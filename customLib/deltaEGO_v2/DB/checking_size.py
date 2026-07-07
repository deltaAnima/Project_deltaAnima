# vibe coding by gemini
import json

# 1. 파일 열기
with open('./VAD.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 길이를 구하고 싶은 키(Key) 이름 입력
# 만약 JSON 전체가 배열 형태라면 바로 len(data)를 쓰면 됩니다.
try:
    # 예: JSON이 {"items": [1, 2, 3]} 구조일 때
    target_key = "items" # <-- 여기에 실제 JSON의 배열 키 이름을 넣으세요.
    length = len(data[target_key])
    
    print(f"배열 '{target_key}'의 길이는: {length}입니다.")

except TypeError:
    # JSON 자체가 리스트인 경우 (예: [1, 2, 3])
    length = len(data)
    print(f"JSON 전체 배열의 길이는: {length}입니다.")

except KeyError:
    print(f"'{target_key}'라는 키를 찾을 수 없습니다. JSON 구조를 확인해보세요!")