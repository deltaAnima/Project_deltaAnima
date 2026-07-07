#include <iostream>
#include <fstream>
#include <vector>
#include <chrono> // 벤치마크용


#include "deltaEGO.hpp" 

int main()
{
    // ====================================================
    // 1. 하드웨어 설정 (Ryzen 9950X CCD Isolation)
    // ====================================================
    PinThreadToCore(16); // CCD 1의 첫 번째 코어로 고정

    // ====================================================
    // 2. 데이터 준비 (Load & Ingest)
    // ====================================================
    std::string json_path = "DB/VAD.json";
    std::ifstream f(json_path);
    if (!f.is_open())
    {
        std::cerr << "[Error] JSON file not found: " << json_path << std::endl;
        // 테스트를 위해 JSON 없으면 더미 데이터라도 만들까요? (선택사항)
        return 1;
    }

    std::cout << "[System] Loading JSON..." << std::endl;
    json j = json::parse(f);

    size_t n = j.size();
    size_t d = 4; // V, A, D + Padding(0)

    // 엔진 생성 (메모리 할당 & 버퍼 준비)
    npcie::Int16Tensor db(n, d);
    
    // 데이터 주입
    db.import_json(j);


    // ====================================================
    // 3. 검색 테스트 (Benchmark)
    // ====================================================
    // 시나리오: 사용자가 "분노(Anger)" 상태임
    // Valence(부정적), Arousal(흥분), Dominance(통제감 높음)
    std::vector<float> query_anger = { -0.8f, 0.9f, 0.7f };

    std::cout << "\n[Test] Searching for 'Anger' (-0.8, 0.9, 0.7)..." << std::endl;

    // 시간 측정 시작
    auto start = std::chrono::high_resolution_clock::now();

    // Top-3 검색
    auto results = db.search_knn(query_anger, 3);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;


    // ====================================================
    // 4. 결과 출력
    // ====================================================
    std::cout << "------------------------------------------------" << std::endl;
    for (size_t i = 0; i < results.size(); ++i)
    {
        std::cout << "Rank " << (i + 1) << ": " 
                  << results[i].first << " (Score: " << results[i].second << ")" << std::endl;
    }
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Search Time: " << elapsed.count() << " ms" << std::endl;

    // L2 캐시 히트율이 높아서 아마 0.00x ms 단위가 나올 겁니다.
    
    return 0;
}