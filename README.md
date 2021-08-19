Generating a description of an image is called image captioning. Image captioning requires to 
recognise the important objects, their attributes and their relationships in an image. It also 
needs to generate syntactically and semantically correct sentences.
Most images do not have a description, but the human can largely understand them without 
their detailed captions. However, machine needs to interpret some form of image captions if 
humans need automatic image captions from it
Image understanding needs to detect and recognize objects. It also needs to understand scene 
type or location, object properties and their interactions. Generating well-formed sentences 
requires both syntactic and semantic understanding of the language.
Inspired by recent work in machine translation and object detection, we introduce an 
attention-based model that automatically learns to describe the content of images. We 
describe how we can train this model in a deterministic manner using standard 
backpropagation techniques and stochastically by maximizing a variational lower bound. We 
also show through visualization how the model is able to automatically learn to fix its gaze on 
salient objects while generating the corresponding words in the output sequence. We validate 
the use of attention with state-of-the art performance on three benchmark datasets: Flickr8k
