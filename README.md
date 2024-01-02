# competition-llm

==summary==

1、baseline模型基于bert用Multiple choice进行训练，参考代码https://huggingface.co/docs/transformers/v4.35.2/tasks/multiple_choice#inference。
2、原始训练数据集只用200个，为了扩充数据集用gpt3.5基于维基百科的数据生成数据集。
3、更改模型结构用