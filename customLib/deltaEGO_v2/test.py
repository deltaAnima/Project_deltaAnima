import delta_ego_core
import json

engine = delta_ego_core.deltaEGO(
    0.8, 0.9, 0.5, 0.7, 0.2,
    0.0, 0.0, 0.0, 1.0
)

if engine.load_vad_db("/home/Reminh/work/deltaAnima/Reminh/VAD.json"):
    print("VAD Database loaded successfully!")
else:
    print("Failed to load VAD.json. Check the file path.")

stimulus_v, stimulus_a, stimulus_d = 0.5, 0.8, 0.3
result_json_str = engine.process_stimulus(stimulus_v, stimulus_a, stimulus_d)
data = json.loads(result_json_str)

print(result_json_str)
print(f"\n[Result Context]")
print(f"Current Emotion: {data['emotion_term']}")
print(f"Similarity: {data['similarity']:.4f}")
print(f"VAD State: {data['current_state']}")
print(f"Stress Level: {data['analysis']['instant']['stress']:.4f}")
print(f"front: {data['analysis']['front']}")
