start_prompt="""We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.\n\nThe activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.
examples:
{example_shots}
text:
{text}
activations:
{activations}
The main thing this neuron does is is to 
"""
start_prompt_for_logit="""We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.\n\nThe activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.
examples:
{example_shots}
text:
{text}
activations:
{activations}
The main thing this neuron does is is to 
"""
summarizing_prompt="""
Now we have the functions of some LLM layers. We have preliminarily summarized the functions of each layer from the logit of each layer of LLM and now we hope you can extract and refine the functions of LLM and summarize the functions of this layer in a few short words. If this layer has multiple functions, you can use ',' to separate them. You can refer to the following example to answer:
example:
Layer function:the main thing this layerdoes is to focus on phrases related to community\n

Summarized function of the layer:processes community-related phrases

Layer function:the main thing this neuron does is to focus on present tense verbs ending in 'ing'

Summarized function of the layer:processes present-tense '-ing' verbs
We have provided the input prompt for the LLM. You can use this input prompt as a reference to summarize the functionality of each layer:{input}
Then you need to summarize the function information and please pay special attention to the content related to encoding and decoding:
Layer function:{info}
Summarized function of the layer:
"""