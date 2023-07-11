# On Enriching Image Captions by Fine-Tuning Large Vision-Language Models with Caption Rewrites

![image](https://github.com/Anonymousanoy/FOHE/assets/137591635/2f2389d4-832e-4fd7-ab71-5a5bbe074c88)

#### Illustration of image caption rewriting using ChatGPT.
In Stage 1, the Keyword Extraction Prompt instructs ChatGPT to generate verbs, nouns, and adjectives (highlighted in brown) from the original caption.
In Stage 2, the Caption Generation Prompt guides ChatGPT to generate a rewritten caption.
By iteratively applying this prompt, multiple rewritten captions can be generated.

```
python gen_augdata.py
```

#### Use different model
You need to first deploy the following models locally.
```
python {use_llava/owl/minigpt4}.py
```

#### Generated example

![image](https://github.com/Anonymousanoy/FOHE/assets/137591635/e3d92dae-bd98-40f6-8350-f76ad965793c)

#### Result

![image](https://github.com/Anonymousanoy/FOHE/assets/137591635/432f953e-194c-4f30-8692-c5d2e6a62d9d)

