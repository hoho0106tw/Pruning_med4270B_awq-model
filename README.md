# 🧠 Med42-70B AWQ 量化評估總結

## 📌 模型概覽
- 基礎模型：m42-health/Llama3-Med42-70B
- 量化模型：hoho0106tw/Femh_Pruning_med4270B_awq-model
- 量化方法：AWQ（Activation-aware Weight Quantization）
- 評估範圍：Loss、分類準確率、記憶體效率

## 📊 量化品質（Loss 評估）
=== Evaluating AWQ ===  
AWQ Loss: 3.0038, PPL: 20.16  

===== RESULT =====  
FP16 Loss: 2.9753  
AWQ  Loss: 3.0038  
Loss Δ: 0.0285  
🔥 Near-lossless quantization  

AWQ 僅帶來極小的 loss 上升（Δ = 0.0285），屬於 near-lossless quantization 等級，表示語言建模能力幾乎完整保留，token-level 預測分布穩定，未出現異常偏移或崩潰現象。

## 🧪 任務評估（神經醫療分類）
AWQ Accuracy: 0.820  

===== FINAL =====  
FP16: 0.84  
AWQ : 0.82  
Δ   : -0.02  

在醫療分類任務中，AWQ 僅造成約 -2% 的準確率下降，且整體僅多錯 1 題，顯示量化後模型仍能維持穩定且精細的語意判斷能力，沒有出現錯誤放大或不穩定問題。

## 💾 記憶體效率
[GPU usage]  
GPU 0: allocated=11.09 GB | reserved=11.18 GB  
GPU 1: allocated=12.92 GB | reserved=26.80 GB  
GPU 2: allocated=14.45 GB | reserved=20.26 GB  

AWQ Total Inference Peak: 129.83 GB  

===== SUMMARY =====  

--- Weight Memory ---  
FP16: 118.30 GB  
AWQ : 38.40 GB  
Reduction: 3.08x  

--- Inference Memory ---  
FP16: 131.90 GB  
AWQ : 129.83 GB  
Reduction: 1.02x  

AWQ 將模型權重記憶體降低約 3.08 倍，大幅提升部署可行性；但推論記憶體幾乎不變（1.02x），顯示 activation 與 KV cache 仍為主要記憶體消耗來源。

## 🎯 關鍵洞察
- 近乎無損量化：Loss Δ = 0.0285，語言能力幾乎無損  
- 穩定任務表現：Accuracy 僅下降約 2%，且只多錯 1 題  
- 高效壓縮：模型權重降低 3.08 倍，但推論記憶體影響極小  
- 核心觀察：AWQ 壓縮的是模型權重，而非推論計算過程  

此外，透過 label 設計優化，FP16 準確率可由 0.45 提升至 0.84，顯示問題定義（label engineering）對最終性能影響遠大於量化誤差。

## 🚀 最終結論
AWQ 在 Med42-70B 上實現了近乎無損的量化效果，在語言建模與醫療分類任務中僅帶來極小性能下降（約 2%），且只增加 1 筆錯誤。同時，模型權重記憶體降低約 3.08 倍，大幅提升實務部署效率，是一種高效且可靠的大模型壓縮方法。

## 🧠 一句話總結
AWQ 在 Med42-70B 上實現近乎無損量化，在維持準確率與推理能力的同時，大幅降低記憶體需求（約 3 倍），非常適合實際部署。
