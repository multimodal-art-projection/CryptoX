logitlens_prompt = """Answer the following multiple choice question. The last line of your response should be one of ABCD. Remember,just give your answer and don't output any other thing.
example:The following paragraphs each describe a set of seven objects arranged in a fixed order. The statements are logically consistent within each paragraph. In an antique car show, there are seven vehicles: a limousine, a truck, a minivan, a tractor, a hatchback, a bus, and a convertible. The convertible is the second-oldest. The tractor is the fourth-newest. The minivan is older than the truck. The truck is older than the limousine. The hatchback is the third-oldest. The bus is newer than the limousine.
A The limousine is the oldest
B The truck is the oldest
C The minivan is the oldest
D The tractor is the oldest
Answer:C
Which of the following is a humorous edit of this artist or movie name: 'the third man'?
A the third men
B the trird man
C thed third man
D the third pan
Answer:D
{rule}

{question}

A {choices_a}
B {choices_b}
C {choices_c}
D {choices_d}

Answer:"""
