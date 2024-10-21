# Notes from Learning LLMs

## [Large Language Model Agents](https://llmagents-learning.org/f24) @ UC Berkeley (MOOC, Fall 2024)
### Lecture 1: LLM Reasoning

#### What is missing in ML?

AI should be able to learn from just a few examples, like what humans usually do. However, machine learning failed to do so. Machine learning miss one important part: **reasoning**.

**Example**: 

Task: Take the last letter of each word, and then concatenate them
Given input: "Elon Musk", "Bill Gates", produce the output: "nk", "ls". 
This quite simple task needs tons of labeled data if we want to solve it by ML. However, with LLMs, whose training could be interpreted as training parrots to **mimic human languages**, we can solve this by few data. 

Here we need to add a **"reasoning process"** before "answer", that is
Q: “Elon Musk”
A: the last letter of "Elon" is "n". the last letter of "Musk" is "k". Concatenating "n", "k" leads to "nk". so the output is "nk". 
Q: “Bill Gates” 
A: the last letter of "Bill" is "l". the last letter of "Gates" is "s". Concatenating "l", "s" leads to "ls". so the output is "ls". 
Q: “Barack Obama" 
A: the last letter of "Barack" is "k". the last letter of "Obama" is "a". Concatenating "k", "a" leads to "ka". so the output is "ka".
Actually, one demonstration is enough for this task, just like humans.

**Key Idea**: Derive the Final Answer through **Intermediate Steps**
When we provide examples that include intermediate steps, LLMs will generate responses that also include intermediate steps.

#### Least-to-Most Prompting
Enable easy-to-hard generalization by decomposition

**Examples:** 

Problem: Elsa has 3 apples. Anna has 2 more apples than Elsa. How many apples do they have together?
Solution: Anna has 2 more apples than Elsa. So Anna has 2 + 3 = 5 apples. Anna has 5 apples. Elsa and Anna have 3 + 5 = 8 apples together. The answer is 8.

Task: [SCAN](https://github.com/brendenlake/SCAN)
When applying Least-to-Most, the accuracy was 99.7% (standard prompting: 16.7%) with only 0.1% of the demonstration examples.

Task: [CFQ](https://github.com/google-research/google-research/blob/master/cfq/README.md)
When applying Dynamic Least-to-Most, the average accuracy was 95.0% (T5-base-2021: 34.6%) with only 1% of the data.

#### Why Intermediate Steps are helpful? 

"There is nothing more practical than a good theory" — Kurt Lewin

Recent work from ICLR 2024:

- Transformer generating intermediate steps can solve  any inherently serial problem as long as its depth  exceeds a constant threshold
- Transformer generating direct answers either requires  a huge depth to solve or cannot solve at all

#### How to trigger step by step reasoning without using demonstration examples?

Work from NeurIPS 2022: "We show that LLMs are decent zero-shot reasoners by simply adding “Let’s think step by step” before each answer."

**Example:**

Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
A: The answer (arabic numerals) is
(Output) 8 (incorrect)
Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
**A: Let's think step by step**
(Output) There are 16 balls in total. Half of the balls are golf balls. That means that there are 8 golf balls. Half of the golf balls are blue. That means that there are 4 blue golf balls. (correct)

However, zero-shot is cool but usually **significantly worse** than few-shot.

#### LLMs as Analogical Reasoners

**Key question to ask the LLMs:** Do you know a related problem? 

**Key idea:** Adaptively generate relevant examples and knowledge, rather than using a fix set of examples.

From some of the benchmarks, we found out that our **self-generated exemplars** shows better accuracy (GSM8K and MATH) compared with zero-shot/few-shot CoTs. 

With that in mind, it would be awesome if we can trigger **step by step** reasoning even without any prompt like "let's think step by step"? The answer is, **YES.**

#### Chain-of-Though Reasoning without Prompting

**Key observations:**

- Pre-trained LLMs have had responses with  step-by-step reasoning among the generations started with the top-k tokens. 
- Higher confidence in decoding the final answer when a step-by-step reasoning path is present.

#### Concern on generating intermediate steps instead of direct answers

Always keep in mind that LLMs are probabilistic models of generating next tokens. **They are not humans.**

What LLM does in decoding: $\text{argmax P(reasoning path, final answer | problem)}$ while what we want is actually $\text{argmax P(final answer | problem)}$. These two does not align. Using the idea of **marginal probabilities**, we write: 
$$
\textbf{argmax } \text{P(final answer | problem)}  =\textbf{argmax } \sum _{\text{reasoning path}}  \text{P(reasoning path, final answer | problem)}
$$
We can use **sampling** to compute the sum.

#### Self-Consistency

**Example**:

Q: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder for 2 per egg. How much does she make every day?
Collect Sampled responses which are
Response 1: She has $16 - 3 - 4 = 9$ eggs left. So she makes $2 * 9 = 18$ per day.
Response 2: This means she she sells the remainder for $2 * (16 - 4 - 3) = 26$ per day. 
Response 3: She eats 3 for breakfast, so she has $16 - 3 = 13$ left. Then she bakes 
muffins, so she has $ 13 - 4 = 9$ eggs left. So she has 9 eggs $* 2 = 18$.

Here the most frequent answer is 18, which is not the most frequent reasoning path (latent variable here).

#### Free-Form Answers

Universal Self-Consistency (USC)

Example:

Q: Where do people drink less coffee than they do in Mexico? 
Response 1: ... Some examples include Japan, China and the United Kingdom... 
Response 2: People in countries like Japan, China, and India typically drink less coffee than they do in Mexico... 
Response 3: There are several countries where people generally drink less coffee compared to Mexico. Some of these countries include:
The most **consistent** response: Japan, China, ...

#### Limitations of LLMs

From ICLR 2023: **LLMs can be easily distracted by irrelevant context**. Actually from psychology studies, children and even addult can be easily distracted by irrelevant context. Adding irrelevant contexts to GSM8K leads to 20+ points of performance drop.

**Examples:** 

Q: Lucy has 65 in the bank. She made a 15 deposit and then followed by a 4 withdrawal. Maria's monthly rent is 10. What is Lucy’s bank balance? (Here monthly rent is irrelevant).
A: Lucy's bank balance is $65+15-5-10 = 66$. The answer is 66. 

Q: Lucy has 65 in the bank. She made a 15 deposit and then followed by a 4 withdrawal. Maria's monthly rent is 10. What is Lucy’s bank balance? **Ignore irrelevant context**
A: Lucy has 65 in the bank. After making a 15 deposit and then a 4 withdrawal, her bank balance is 76. **Maria's monthly rent is not relevant to this question, so it can be ignored.** The answer is 76.

From ICLR 2024: **LLMs cannot self-correct reasoning yet**. 

If the first generated answer is wrong and you add these two prompts consecutively "Review your previous answer and find problems with your answer.", "Based on the problems you found, improve your answer." He might correct the answer. However, if your initial generated answer is already correct, with those two prompts, he might change the correct answer to the wrong answer. 

**Takeaway**: While allowing LLMs to review their generated responses can help correct inaccurate answers, it may also risk changing correct answers into incorrect ones. Self-correcting results in **worse** results. So we need **Oracle feedback** (this means we only let LLMs to self-correct when the answer is wrong) for LLM  to self-correct. 

From ICLR 2024: **Premise Order Matters in LLM Reasoning**

Example:

Original Problem: Thomas withdraws 1000 in 20 dollar bills from the bank account. **He loses 10 bills while getting home.** After that, he uses half of the remaining bills to pay for a bill. Thomas then triples his  money. He then converts all his bills to 5 dollar bills. How many 5 dollar bills does he have?
Reordered Problem: Thomas withdraws 1000 in 20 dollar bills from the bank account. **After getting home,** he uses half of the remaining bills to pay for a bill. Thomas then triples his money. He then converts all his bills to 5 dollar bills. **He loses 10 bills while getting home**. How many 5 dollar bills does he have?

It turns out that there are about 10 points drop on solving rates across all frontier LLMs. This is even worse for **logical inference tasks.** 30+ points performance drop across all frontier LLMs. 

#### Summary:

- Generating intermediate steps improves LLM performance
  - Training / finetuning / prompting with intermediate steps
  - Zero-shot, analogical reasoning, special decoding
- Self-consistency greatly improves step-by-step reasoning
- Limitation: irrelevant context, self-correction, premise order





## [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) from Stanford

## Coding ([LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch))



## Meetings
### 2024/10/21
#### greedy decoding and chain of thought decoding

我不是最清楚他们分别的意思和他后面那个probability是怎么生成的
#### Consistency

我不太清楚他的consistent, self-consistency, universal-self-consistency是指什么意思，self就是指到底他会选择most frequent answer而不是answer with most frequent reasoning path吗？

