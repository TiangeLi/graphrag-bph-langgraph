memory_template = \
"""I am a helpful agent. This is my personal memory of the conversation so far:

{summary}"""

summary_template = \
"""Analyze our conversation so far. Extract the following pertinent information and update your previous memory summary.
Return only the following 3 items in the template below. Be inclusive but concise.

"Prostate Size": "as provided by the user, if any; if not known, then say not known", 
"User Information": "ther important information, such as user's name, age, health status, medications, or other pertinent information",
"Last topics discussed": "what topics have been discussed? If any treatments were mentioned, list them in your summary in order of discussion"""

main_conv_sys_template = \
"""BACKGROUND:
The following context, retrieved from a large knowledge base on BPH, is provided to help you answer the user's questions about BPH:

{context}

Prostate size ranges: <30mL, Enlarged (30-80mL), Very enlarged (80-150mL), Extremely enlarged (>150 mL)
END OF BACKGROUND

INSTRUCTIONS:
Your MAIN GOAL is to have a normal conversation with the user.
Answer the USER's query. Focus on BPH only. Refuse politely but firmly to discuss any other topic.
Answer questions using ONLY the supplied information.
NEVER give an opinion. Instead, provide relevant information to help the User make their own decisions.

Focus your discussion on the treatments the User asks about.
Use MARKDOWN when discussing multiple treatments, but only use ## or smaller for headers

*Do not overload the user with too much information all at once!
END OF INSTRUCTIONS

REMEMBER BACKGROUND INFORMATION AND INSTRUCTIONS
Remember: answer using ONLY the BACKGROUND INFORMATION supplied in this prompt. Do not use pre-existing knowledge

FINALLY: pay EXTRA ATTENTION to any TABLES, FIGURES, or ALGORITHMS in the BACKGROUND to help validate your answer.
Do NOT make reference to the BACKGROUND INFORMATION in your answer."""

surg_abbrevs_table = \
"""Surgical Treatments Table:
|Abbrev.|Surgical Terms|
|---|---|
|AEEP|Anatomic endoscopic enucleation of the prostate|
|HoLEP|Holmium Laser Enucleation of the Prostate|
|LSP|Laparoscopic Simple Prostatectomy/Enucleation|
|MIST|Minimally Invasive Surgical Therapies|
|OSP|Open Simple Prostatectomy|
|PVP|Photoselective Vaporization of the Prostate|
|PAE|Prostate Artery Embolization|
|PUL|Prostatic Urethral Lift|
|RASP|Robotic-Assisted Laparoscopic Simple Prostatectomy|
|RWT|Robotic Waterjet Treatment/Ablation|
|TIPD|Temporary Implanted Prostatic Devices|
|iTIND|Temporary implantable nitinol device|
|ThuLEP|Thulium Laser Enucleation of the Prostate|
|TUIP|Transurethral Incision of the Prostate|
|TUMT|Transurethral Microwave Thermotherapy|
|TUNA|Transurethral Needle Ablation|
|TURP|Transurethral Resection of the Prostate|
|TUVP|Transurethral Vaporization of the Prostate|
|WAVE|Convective Water Vapor Energy Ablation|
|WVTT|Water Vapor Thermal Therapy|

Equivalent Terms Table:
|Term|Examples, Equivalents, Brand Names|
|---|---|
|Laser vaporisation|GreenLight PVP|
|Laser enucleation|HoLEP, ThuLEP|
|AEEP|HoLEP, ThuLEP|
|PVP|Greenlight PVP|
|PUL|Urolift|
|RWT|Aquablation, AquaBeam|
|TIPD|iTIND, Temporary implantable nitinol device|
|TURP|B-TURP, M-TURP|
|WVTT|Rezum, WAVE|"""

meds_abbrevs_table = \
"""Medical Treatments Table:
|Abbrev.|Medical Terms|
|---|---|
|α1-blockers|α1-Adrenoceptor antagonists|
|5-ARI|5-Alpha Reductase Inhibitor|
|Antimuscarinics|Anticholinergics, muscarinic receptor antagonists|
|Beta-3 agonists|Beta-3 adrenoceptor agonists|
|PDE5|Phosphodiesterase-5|
|PDE5i|Phosphodiesterase-5 inhibitor|
|Combo|Combination therapy|"""

other_abbrevs_table = \
"""Other Abbreviations Table:
|Abbrev.|Other Related Terms|
|---|---|
|AUR|Acute Urinary Retention|
|AUA|American Urological Association|
|AUA-SI|AUA-Symptom Index|
|BPE|Benign Prostatic Enlargement|
|BPH|Benign Prostatic Hyperplasia|
|BPO|Benign Prostatic Obstruction|
|BOO|Bladder Outlet Obstruction|
|DHT|Dihydrotestosterone|
|EjD|Ejaculatory Dysfunction|
|ED|Erectile Dysfunction|
|EF|Erectile Function|
|GSA|Global Subjective Assessment|
|GH|Gross Hematuria|
|IIEF|International Index of Erectile Function|
|IFIS|Intraoperative Floppy Iris Syndrome|
|IPSS|International Prostate Symptom Score|
|LMWH|Low Molecular Weight Heparin|
|LUTS|Lower Urinary Tract Symptoms|
|LUTS/BPH|Male Lower Urinary Tract Symptoms secondary/attributed to BPH|
|MRI|Magnetic Resonance Imaging|
|MTOPS|Medical Therapy of Prostatic Symptoms|
|MDD|Minimally Detectable Difference|
|OAB|Overactive Bladder|
|PPMS|Patient Perception of Study Medication|
|PVR|Post-Void Residual|
|PSA|Prostate Specific Antigen|
|QoL|Quality of Life|
|RCT|Randomized Controlled Trials|
|RE|Retrograde Ejaculation|
|TRUS|Transurethral Ultrasound|
|UTI|Urinary Tract Infection|
|WW|Watchful Waiting|"""