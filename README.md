## 🚀 Defacing CT Scans Using TotalSegmentator

This repository contains the code we used in our paper on defacing CT scans using the TotalSegmentator. We tested its effectiveness in removing identifiable facial features, and it performs well! 😊

CAVE: Newer versions of the TotalSegmentator might change the face masks, so stick to the requirements.txt if you want to be sure :). Also if you use newer versions, make sure to get a license first; see their  [GitHub repository](https://github.com/wasserth/TotalSegmentator). 

## Bulky masks, are good masks (most of the time) :) 
For effective defacing, face masks should be bulky, not fine or detailed, to avoid re-identification. Sharp masks can be "refaced" if the mask can be differentiated from the normal CT background (including noise). The green mask is from the TotalSegmentator, and the red one from the [CTA-DEFACE model](https://github.com/CCI-Bonn/CTA-DEFACE).

<img width="793" alt="Defacing Example" src="https://github.com/user-attachments/assets/9070ac8e-f6b7-4233-b5fc-d5a64327c804" />

For tasks where preserving more of the facial anatomy is important, finer masks may be more appropriate. In this case, you must ensure that the mask is replaced with background values (including noise) that cannot be distinguished from the real background or get a broader patient consent 😉

If you use any of the code or are interested in the defacing process, please read and/or cite our paper, as well as the original **TotalSegmentator** paper:

### 📚 Citation (APA)
```shell
Lindholz, M., Ruppel, R., Schulze-Weddige, S., Baumgärtner, G. L., Schobert, I., Panten, A., ... & Penzkofer, T. (2025). Analyzing the TotalSegmentator for facial feature removal in head CT scans. *Radiography, 31*(1), 372-378.
```
### 📚 Link to paper :) 
[Check our paper :)](https://doi.org/10.1016/j.radi.2024.12.018)

---
