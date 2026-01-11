import sys
import argparse
import time
import numpy as np
import sounddevice as sd
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
