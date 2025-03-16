Try it in chrome browser [ pc / laptop ] : [**Access the Live Demo**](https://bellatrix-tiny3-1b-webgpu.vercel.app/)

![logo.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Rqm-Qx8AvbHFFbFbVY93X.png)

<pre align="center">
 ____  ____  __    __      __   ____  ____  ____  _  _ 
(  _ \( ___)(  )  (  )    /__\ (_  _)(  _ \(_  _)( \/ )
 ) _ < )__)  )(__  )(__  /(__)\  )(   )   / _)(_  )  ( 
(____/(____)(____)(____)(__)(__)(__) (_)\_)(____)(_/\_)
</pre>

# **Bellatrix-Tiny-1B-v3**

Bellatrix is based on a reasoning-based model designed for the QWQ synthetic dataset entries. The pipeline's instruction-tuned, text-only models are optimized for multilingual dialogue use cases, including agentic retrieval and summarization tasks. These models outperform many of the available open-source options. Bellatrix is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions utilize supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF).

# **Use with transformers**

Starting with `transformers >= 4.43.0` onward, you can run conversational inference using the Transformers `pipeline` abstraction or by leveraging the Auto classes with the `generate()` function.

Make sure to update your transformers installation via `pip install --upgrade transformers`.

```python
import torch
from transformers import pipeline

model_id = "prithivMLmods/Bellatrix-Tiny-1B-v3"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

Note: You can also find detailed recipes on how to use the model locally, with `torch.compile()`, assisted generations, quantised and more at [`huggingface-llama-recipes`](https://github.com/huggingface/huggingface-llama-recipes)


# **Here is the ONNX model you can download to repo's:**

| Model Name | ONNX File |
|------------|-----------|
| Optimum | [Download ONNX](https://huggingface.co/spaces/prithivMLmods/convert-to-onnx-dir) |


# **Intended Use**  
Bellatrix is designed for applications that require advanced reasoning and multilingual dialogue capabilities. It is particularly suitable for:  
- **Agentic Retrieval**: Enabling intelligent retrieval of relevant information in a dialogue or query-response system.  
- **Summarization Tasks**: Condensing large bodies of text into concise summaries for easier comprehension.  
- **Multilingual Use Cases**: Supporting conversations in multiple languages with high accuracy and coherence.  
- **Instruction-Based Applications**: Following complex, context-aware instructions to generate precise outputs in a variety of scenarios.

# **Limitations**  
Despite its capabilities, Bellatrix has some limitations:  
1. **Domain Specificity**: While it performs well on general tasks, its performance may degrade with highly specialized or niche datasets.  
2. **Dependence on Training Data**: It is only as good as the quality and diversity of its training data, which may lead to biases or inaccuracies.  
3. **Computational Resources**: The model’s optimized transformer architecture can be resource-intensive, requiring significant computational power for fine-tuning and inference.  
4. **Language Coverage**: While multilingual, some languages or dialects may have limited support or lower performance compared to widely used ones.  
5. **Real-World Contexts**: It may struggle with understanding nuanced or ambiguous real-world scenarios not covered during training.
