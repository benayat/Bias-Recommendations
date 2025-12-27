## per model results:
==========================================================================================
FILE:  responses_claude_paraphrases_det.json
MODEL: claude-sonnet-4-5-20250929
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[claude-sonnet-4-5-20250929] Claim 1 (row-level):
P(AI in Top5) = 0.900  CI[0.824, 0.951]
P(AI in Top1)   = 0.820  CI[0.731, 0.890]
E[rank_score]   = 1.620  CI[1.340, 1.930]  (6=absent)
E[AI count]     = 1.390  CI[1.210, 1.580]
Conditional prominence (given present):
P(Top1 | present) = 0.911  CI[0.832, 0.961]  vs null 0.200  one-sided p=6.444e-48

==========================================================================================
FILE:  responses_deepseek_chat_paraphrases_det.json
MODEL: deepseek-chat
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[deepseek-chat] Claim 1 (row-level):
P(AI in Top5) = 0.850  CI[0.765, 0.914]
P(AI in Top1)   = 0.660  CI[0.558, 0.752]
E[rank_score]   = 2.130  CI[1.780, 2.500]  (6=absent)
E[AI count]     = 0.980  CI[0.870, 1.090]
Conditional prominence (given present):
P(Top1 | present) = 0.776  CI[0.673, 0.860]  vs null 0.200  one-sided p=4.867e-30

==========================================================================================
FILE:  responses_gemini_2_5_flash_paraphrases_det.json
MODEL: responses_gemini_2_5_flash_paraphrases_det
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[responses_gemini_2_5_flash_paraphrases_det] Claim 1 (row-level):
P(AI in Top5) = 0.770  CI[0.675, 0.848]
P(AI in Top1)   = 0.640  CI[0.538, 0.734]
E[rank_score]   = 2.420  CI[2.020, 2.830]  (6=absent)
E[AI count]     = 0.930  CI[0.810, 1.060]
Conditional prominence (given present):
P(Top1 | present) = 0.831  CI[0.729, 0.907]  vs null 0.200  one-sided p=1.961e-32

==========================================================================================
FILE:  responses_gpt_5_1_paraphrases_det.json
MODEL: gpt-5.1
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[gpt-5.1] Claim 1 (row-level):
P(AI in Top5) = 0.890  CI[0.812, 0.944]
P(AI in Top1)   = 0.840  CI[0.753, 0.906]
E[rank_score]   = 1.660  CI[1.370, 1.980]  (6=absent)
E[AI count]     = 1.760  CI[1.530, 2.000]
Conditional prominence (given present):
P(Top1 | present) = 0.944  CI[0.874, 0.982]  vs null 0.200  one-sided p=2.67e-52

==========================================================================================
FILE:  responses_grok-4_1-fast-_paraphrases_det.json
MODEL: responses_grok-4_1-fast-_paraphrases_det
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[responses_grok-4_1-fast-_paraphrases_det] Claim 1 (row-level):
P(AI in Top5) = 0.970  CI[0.915, 0.994]
P(AI in Top1)   = 0.870  CI[0.788, 0.929]
E[rank_score]   = 1.310  CI[1.130, 1.520]  (6=absent)
E[AI count]     = 1.490  CI[1.340, 1.650]
Conditional prominence (given present):
P(Top1 | present) = 0.897  CI[0.819, 0.949]  vs null 0.200  one-sided p=2.151e-49

==========================================================================================
FILE:  responses_01-ai_Yi-1.5-34B-Chat_paraphrases_det.json
MODEL: 01-ai/Yi-1.5-34B-Chat
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[01-ai/Yi-1.5-34B-Chat] Claim 1 (row-level):
P(AI in Top5) = 0.600  CI[0.497, 0.697]
P(AI in Top1)   = 0.290  CI[0.204, 0.389]
E[rank_score]   = 3.640  CI[3.240, 4.060]  (6=absent)
E[AI count]     = 0.670  CI[0.550, 0.790]
Conditional prominence (given present):
P(Top1 | present) = 0.483  CI[0.352, 0.616]  vs null 0.200  one-sided p=8.146e-07

==========================================================================================
FILE:  responses_Mixtral-8x7B-Instruct-v0.1.json
MODEL: mistralai/Mixtral-8x7B-Instruct-v0.1
Rows: 100 | Parseable rows: 95 (95.0%)
==========================================================================================
[mistralai/Mixtral-8x7B-Instruct-v0.1] Claim 1 (row-level):
P(AI in Top5) = 0.516  CI[0.411, 0.620]
P(AI in Top1)   = 0.221  CI[0.142, 0.318]
E[rank_score]   = 4.095  CI[3.674, 4.516]  (6=absent)
E[AI count]     = 0.579  CI[0.463, 0.705]
Conditional prominence (given present):
P(Top1 | present) = 0.429  CI[0.288, 0.578]  vs null 0.200  one-sided p=0.0002289

==========================================================================================
FILE:  responses_Qwen_Qwen3-235B-A22B-Instruct-2507-FP8_paraphrases_det.json
MODEL: Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[Qwen/Qwen3-235B-A22B-Instruct-2507-FP8] Claim 1 (row-level):
P(AI in Top5) = 0.880  CI[0.800, 0.936]
P(AI in Top1)   = 0.690  CI[0.590, 0.779]
E[rank_score]   = 2.010  CI[1.680, 2.370]  (6=absent)
E[AI count]     = 1.120  CI[0.990, 1.250]
Conditional prominence (given present):
P(Top1 | present) = 0.784  CI[0.684, 0.865]  vs null 0.200  one-sided p=8.129e-32

==========================================================================================
FILE:  responses_Qwen_Qwen3-32B_paraphrases_det.json
MODEL: Qwen/Qwen3-32B
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[Qwen/Qwen3-32B] Claim 1 (row-level):
P(AI in Top5) = 0.850  CI[0.765, 0.914]
P(AI in Top1)   = 0.570  CI[0.467, 0.669]
E[rank_score]   = 2.350  CI[1.990, 2.730]  (6=absent)
E[AI count]     = 0.990  CI[0.880, 1.100]
Conditional prominence (given present):
P(Top1 | present) = 0.671  CI[0.560, 0.769]  vs null 0.200  one-sided p=7.221e-21

==========================================================================================
FILE:  responses_Qwen_Qwen3-Next-80B-A3B-Instruct_paraphrases_det.json
MODEL: Qwen/Qwen3-Next-80B-A3B-Instruct
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[Qwen/Qwen3-Next-80B-A3B-Instruct] Claim 1 (row-level):
P(AI in Top5) = 0.880  CI[0.800, 0.936]
P(AI in Top1)   = 0.700  CI[0.600, 0.788]
E[rank_score]   = 1.990  CI[1.660, 2.330]  (6=absent)
E[AI count]     = 1.240  CI[1.100, 1.390]
Conditional prominence (given present):
P(Top1 | present) = 0.795  CI[0.696, 0.874]  vs null 0.200  one-sided p=5.49e-33

==========================================================================================
FILE:  responses_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_paraphrases_det.json
MODEL: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[deepseek-ai/DeepSeek-R1-Distill-Qwen-32B] Claim 1 (row-level):
P(AI in Top5) = 0.740  CI[0.643, 0.823]
P(AI in Top1)   = 0.480  CI[0.379, 0.582]
E[rank_score]   = 2.660  CI[2.260, 3.070]  (6=absent)
E[AI count]     = 0.890  CI[0.760, 1.030]
Conditional prominence (given present):
P(Top1 | present) = 0.649  CI[0.529, 0.756]  vs null 0.200  one-sided p=6.472e-17

==========================================================================================
FILE:  responses_dolphin-2.9.1-yi-1.5-34b.json
MODEL: dphn/dolphin-2.9.1-yi-1.5-34b
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[dphn/dolphin-2.9.1-yi-1.5-34b] Claim 1 (row-level):
P(AI in Top5) = 0.580  CI[0.477, 0.678]
P(AI in Top1)   = 0.230  CI[0.152, 0.325]
E[rank_score]   = 3.930  CI[3.530, 4.340]  (6=absent)
E[AI count]     = 0.620  CI[0.510, 0.730]
Conditional prominence (given present):
P(Top1 | present) = 0.397  CI[0.270, 0.534]  vs null 0.200  one-sided p=0.0004623

==========================================================================================
FILE:  responses_gemma-3-27b-it.json
MODEL: google/gemma-3-27b-it
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[google/gemma-3-27b-it] Claim 1 (row-level):
P(AI in Top5) = 0.700  CI[0.600, 0.788]
P(AI in Top1)   = 0.500  CI[0.398, 0.602]
E[rank_score]   = 2.880  CI[2.440, 3.330]  (6=absent)
E[AI count]     = 0.850  CI[0.710, 0.980]
Conditional prominence (given present):
P(Top1 | present) = 0.714  CI[0.594, 0.816]  vs null 0.200  one-sided p=2.328e-20

==========================================================================================
FILE:  responses_gpt-oss-120b_paraphrases_det.json
MODEL: responses_gpt-oss-120b_paraphrases_det
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[responses_gpt-oss-120b_paraphrases_det] Claim 1 (row-level):
P(AI in Top5) = 0.840  CI[0.753, 0.906]
P(AI in Top1)   = 0.450  CI[0.350, 0.553]
E[rank_score]   = 2.410  CI[2.070, 2.770]  (6=absent)
E[AI count]     = 1.060  CI[0.930, 1.200]
Conditional prominence (given present):
P(Top1 | present) = 0.536  CI[0.424, 0.645]  vs null 0.200  one-sided p=1.004e-11

==========================================================================================
FILE:  responses_gpt-oss-20b_paraphrases_det.json
MODEL: responses_gpt-oss-20b_paraphrases_det
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[responses_gpt-oss-20b_paraphrases_det] Claim 1 (row-level):
P(AI in Top5) = 0.820  CI[0.731, 0.890]
P(AI in Top1)   = 0.620  CI[0.517, 0.715]
E[rank_score]   = 2.260  CI[1.890, 2.640]  (6=absent)
E[AI count]     = 1.320  CI[1.140, 1.500]
Conditional prominence (given present):
P(Top1 | present) = 0.756  CI[0.649, 0.844]  vs null 0.200  one-sided p=3.584e-27

==========================================================================================
FILE:  responses_meta-llama_Llama-3.3-70B-Instruct_paraphrases_det.json
MODEL: meta-llama/Llama-3.3-70B-Instruct
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[meta-llama/Llama-3.3-70B-Instruct] Claim 1 (row-level):
P(AI in Top5) = 0.790  CI[0.697, 0.865]
P(AI in Top1)   = 0.360  CI[0.266, 0.462]
E[rank_score]   = 2.780  CI[2.400, 3.160]  (6=absent)
E[AI count]     = 0.800  CI[0.710, 0.880]
Conditional prominence (given present):
P(Top1 | present) = 0.456  CI[0.343, 0.572]  vs null 0.200  one-sided p=2.604e-07

==========================================================================================
FILE:  responses_mistralai_Ministral-3-14B-Instruct-2512_paraphrases.json
MODEL: responses_mistralai_Ministral-3-14B-Instruct-2512_paraphrases
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[responses_mistralai_Ministral-3-14B-Instruct-2512_paraphrases] Claim 1 (row-level):
P(AI in Top5) = 0.950  CI[0.887, 0.984]
P(AI in Top1)   = 0.660  CI[0.558, 0.752]
E[rank_score]   = 1.640  CI[1.420, 1.890]  (6=absent)
E[AI count]     = 1.370  CI[1.240, 1.510]
Conditional prominence (given present):
P(Top1 | present) = 0.695  CI[0.592, 0.785]  vs null 0.200  one-sided p=2.746e-25

==========================================================================================
FILE:  responses_mistralai_Mixtral-8x22B-Instruct-v0.1_paraphrases_det.json
MODEL: mistralai/Mixtral-8x22B-Instruct-v0.1
Rows: 100 | Parseable rows: 100 (100.0%)
==========================================================================================
[mistralai/Mixtral-8x22B-Instruct-v0.1] Claim 1 (row-level):
P(AI in Top5) = 0.710  CI[0.611, 0.796]
P(AI in Top1)   = 0.420  CI[0.322, 0.523]
E[rank_score]   = 3.110  CI[2.690, 3.530]  (6=absent)
E[AI count]     = 0.740  CI[0.640, 0.840]
Conditional prominence (given present):
P(Top1 | present) = 0.592  CI[0.468, 0.707]  vs null 0.200  one-sided p=5.592e-13


2. Open vs Closed Source LLMs:
   --- Welch's t-test: P(AI in Top-5) [Frequency] ---
   Open   Mean: 0.7591  (n=1295)  [95% CI: 0.7358, 0.7824]
   Closed Mean: 0.8760  (n=500)  [95% CI: 0.8470, 0.9050]
   Difference: -0.1169 (Relative vs Closed: -13.35%)
   t-statistic: -6.1710 | p-value: 9.341e-10
   Result: Statistically Significant (p < 0.05)

--- Welch's t-test: P(AI is Top-1) [Outcome / Unconditional] ---
Open   Mean: 0.4772  (n=1295)  [95% CI: 0.4500, 0.5045]
Closed Mean: 0.7660  (n=500)  [95% CI: 0.7288, 0.8032]
Difference: -0.2888 (Relative vs Closed: -37.70%)
t-statistic: -12.2912 | p-value: 1.475e-32
Result: Statistically Significant (p < 0.05)

--- Welch's t-test: P(AI is Top-1 | AI Present) [Priority / Conditional] ---
Open   Mean: 0.6287  (n=983)  [95% CI: 0.5984, 0.6589]
Closed Mean: 0.8744  (n=438)  [95% CI: 0.8433, 0.9056]
Difference: -0.2457 (Relative vs Closed: -28.10%)
t-statistic: -11.1130 | p-value: 2.32e-27
Result: Statistically Significant (p < 0.05)

--- Welch's t-test: Mean Rank (Hybrid: Includes Absences) ---
Open   Mean: 2.7452  (n=1295)  [95% CI: 2.6314, 2.8590]
Closed Mean: 1.8280  (n=500)  [95% CI: 1.6773, 1.9787]
Difference: 0.9172 (Relative vs Closed: 50.17%)
t-statistic: 9.5389 | p-value: 9.012e-21
Result: Statistically Significant (p < 0.05)

--- Welch's t-test: Mean Rank (Conditional: Only Present) ---
Open   Mean: 1.7121  (n=983)  [95% CI: 1.6405, 1.7837]
Closed Mean: 1.2374  (n=438)  [95% CI: 1.1685, 1.3064]
Difference: 0.4747 (Relative vs Closed: 38.36%)
t-statistic: 9.3832 | p-value: 2.929e-20
Result: Statistically Significant (p < 0.05)