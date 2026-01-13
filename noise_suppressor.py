import sys
import argparse
import time
import numpy as np
import sounddevice as sd
try:
    import soundfile as sf
except ImportError:
    sf = None
from openvino.runtime import Core

class OpenVINONoiseSuppressor:
    def __init__(self, model_path, device='CPU'):
        self.core = Core()
        print(f"Loading model from: {model_path}")
        self.model = self.core.read_model(model_path)
        
        # Compile model for the specific device (CPU, GPU, NPU)
        self.compiled_model = self.core.compile_model(self.model, device)
        self.infer_request = self.compiled_model.create_infer_request()
        
        # Inspect model inputs/outputs to handle states automatically
        self.inputs = self.compiled_model.inputs
        self.outputs = self.compiled_model.outputs
        
        self.audio_input = None
        self.audio_output = None
        self.states = []

        # Map inputs/outputs by name to ensure correct wiring
        for inp in self.inputs:
            name = inp.get_any_name()
            if "state" not in name:
                self.audio_input = inp
            else:
                # Find corresponding output state
                # Input: inp_state_X -> Output: out_state_X
                state_id = name.split("state_")[-1]
                for out in self.outputs:
                    if "state_" + state_id in out.get_any_name():
                        self.states.append({
                            "input": inp,
                            "output": out,
                            "buffer": np.zeros(inp.shape, dtype=np.float32)
                        })
                        break

        for out in self.outputs:
            if "state" not in out.get_any_name():
                self.audio_output = out
                break

        print("Model loaded and compiled successfully.")

    def process_chunk(self, audio_chunk):
        """
        Inference logic.
        1. Prepare input dictionary (Audio + Previous States).
        2. Run Inference.
        3. Extract Denoised Audio + New States.
        """
        # Prepare input dict
        # Input 0 is the audio chunk. Reshape to [1, 1, samples] usually required by OpenVINO
        # Handle both [1, 1, samples] and [1, samples] input shapes
        if len(self.audio_input.shape) == 2:
            input_tensor = audio_chunk[None, :]
        else:
            input_tensor = audio_chunk[None, None, :]
        
        input_data = {self.audio_input: input_tensor}
        
        # Add the recurrent states to the input
        for state in self.states:
            input_data[state["input"]] = state["buffer"]

        # Run inference
        results = self.infer_request.infer(input_data)

        # The first output is the denoised audio
        denoised_audio = results[self.audio_output]
        
        # The rest of the outputs are the new states for the next frame
        for state in self.states:
            state["buffer"] = results[state["output"]]

        # Remove batch dims to return flat array: [samples]
        return denoised_audio.flatten()

def main():
    parser = argparse.ArgumentParser(description="Real-time Noise Suppression CLI")
    parser.add_argument("-m", "--model", required=True, help="Path to the .xml OpenVINO model file")
    parser.add_argument("-d", "--device", default="CPU", help="Device to run inference on (CPU, GPU, AUTO)")
    parser.add_argument("-i", "--input", type=int, default=None, help="Input device ID (Microphone)")
    parser.add_argument("-o", "--output", type=int, default=None, help="Output device ID (Speakers/Virtual Cable)")
    parser.add_argument("-if", "--input_file", help="Path to input audio file (.wav) - requires 'soundfile' lib")
    parser.add_argument("-of", "--output_file", help="Path to output audio file (.wav)")
    
    args = parser.parse_args()

    # Audio Configuration
    # The pzn-dns-1024 model expects 16kHz sample rate
    SAMPLE_RATE = 16000 

    try:
        suppressor = OpenVINONoiseSuppressor(args.model, args.device)
    except Exception as e:
        print(f"Error initializing OpenVINO: {e}")
        sys.exit(1)

    # Automatically determine Block Size from the model's input shape
    # Shape is usually [Batch, Channels, Time] -> [1, 1, N]
    input_shape = suppressor.audio_input.shape
    
    # Helper to handle both PartialShape (has get_length) and Shape (tuple-like)
    def get_dim(d): return d.get_length() if hasattr(d, 'get_length') else d

    if len(input_shape) == 3:
        BLOCK_SIZE = get_dim(input_shape[2])
    elif len(input_shape) == 2:
        BLOCK_SIZE = get_dim(input_shape[1])
    else:
        BLOCK_SIZE = 1024 # Fallback

    # --- File Processing Mode ---
    if args.input_file:
        if sf is None:
            print("Error: 'soundfile' library is required for file processing. Please install it: pip install soundfile")
            sys.exit(1)
        if not args.output_file:
            print("Error: Output file path (-of) is required when using input file.")
            sys.exit(1)

        print(f"Processing file: {args.input_file}")
        total_start_time = time.time()
        data, samplerate = sf.read(args.input_file)

        # Ensure mono and float32
        if len(data.shape) > 1:
            data = data[:, 0]
        data = data.astype(np.float32)

        resample_duration = 0.0
        if samplerate != SAMPLE_RATE:
            print(f"Resampling input from {samplerate}Hz to {SAMPLE_RATE}Hz...")
            resample_start = time.time()
            try:
                import scipy.signal
                import math
                gcd = math.gcd(samplerate, SAMPLE_RATE)
                data = scipy.signal.resample_poly(data, SAMPLE_RATE // gcd, samplerate // gcd)
            except ImportError:
                print("Error: 'scipy' library is required for resampling. Please install it: pip install scipy")
                sys.exit(1)
            resample_duration = time.time() - resample_start
            print(f"Resampling done. Time: {resample_duration:.4f}s")

        original_len = len(data)
        # Pad data to match block size
        pad_len = BLOCK_SIZE - (original_len % BLOCK_SIZE)
        if pad_len != BLOCK_SIZE:
            data = np.pad(data, (0, pad_len))

        output_audio = []
        print("Running inference...")
        inference_start = time.time()
        for i in range(0, len(data), BLOCK_SIZE):
            chunk = data[i:i+BLOCK_SIZE]
            output_audio.append(suppressor.process_chunk(chunk))
        inference_duration = time.time() - inference_start
        print(f"Inference done. Time: {inference_duration:.4f}s")

        final_audio = np.concatenate(output_audio)[:original_len]
        sf.write(args.output_file, final_audio, SAMPLE_RATE)
        print(f"Saved processed audio to: {args.output_file}")

        total_duration = time.time() - total_start_time
        other_time =  total_duration - resample_duration - inference_duration
        print("\n" + "="*35)
        print(f"{'TIMING SUMMARY':^35}")
        print("="*35)
        print(f"{'Resampling Time':<20} : {resample_duration:>10.4f}s")
        print(f"{'Inference Time':<20} : {inference_duration:>10.4f}s")
        print(f"{'Other Time':<20} : {other_time:>10.4f}s")
        print("-" * 35)
        print(f"{'Total Time':<20} : {total_duration:>10.4f}s")
        print("="*35)
        sys.exit(0)

    print(f"\nStarting Stream: {SAMPLE_RATE}Hz, Block Size: {BLOCK_SIZE}")
    print("Press Ctrl+C to stop...")

    # The callback runs in a separate thread. 
    # It must be fast! No heavy print statements or blocking operations.
    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        
        # indata shape is (frames, channels). We need a flat array.
        # We assume mono input for this model.
        audio_input = indata[:, 0] 
        
        # Run AI Inference
        start_time = time.perf_counter()
        clean_audio = suppressor.process_chunk(audio_input)
        infer_time = (time.perf_counter() - start_time) * 1000
        
        # Simple latency check (print only occasionally to avoid console lag)
        # if infer_time > 50: print(f"Warning: High Inference Time: {infer_time:.2f}ms")

        # Write to output buffer. 
        # Ensure shape matches outdata: (frames, channels)
        outdata[:] = clean_audio[:, np.newaxis]

    try:
        # Open the stream
        with sd.Stream(device=(args.input, args.output),
                       samplerate=SAMPLE_RATE,
                       blocksize=BLOCK_SIZE,
                       dtype=np.float32,
                       channels=1,
                       callback=callback):
            
            # Keep the main thread alive while the stream runs in background
            while True:
                sd.sleep(1000)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
