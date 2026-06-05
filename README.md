# EX-02-Cross-Platform-Prompting-Evaluating-Diverse-Techniques-in-AI-Powered-Text-Summarization

## AIM
To evaluate and compare the effectiveness of prompting techniques (zero-shot, few-shot, chain-of-thought, role-based) across different AI platforms (e.g., ChatGPT, Gemini, Claude, Copilot) in a specific task: text summarization.

## SCENARIO:
You are part of a content curation team for an educational platform that delivers quick summaries of research papers to undergraduate students. Your task is to summarize a 500-word technical article on "The Basics of Blockchain Technology" using multiple AI platforms and prompting strategies.

Your goal is to determine which combination of prompting technique + platform provides the best summary in terms of:

Accuracy

Coherence

Simplicity

Speed

User experience

## OUTPUT:

NAME: SRIRAM.S

REG.NO: 212225240155

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("titanic_dataset.csv")
df
```

<img width="787" height="295" alt="image" src="https://github.com/user-attachments/assets/b152fa43-40f1-4964-b28b-057c86403aa8" />

```
print(df.info())
print()
print(df.dtypes)
```

<img width="394" height="682" alt="image" src="https://github.com/user-attachments/assets/4385e611-8e15-41b9-84c3-83b148c62235" />

```
df.value_counts()
```

<img width="1148" height="499" alt="image" src="https://github.com/user-attachments/assets/2aead5a6-8408-454a-a96b-af9152c6801b" />

```
df['Age'].value_counts()
```

<img width="627" height="479" alt="image" src="https://github.com/user-attachments/assets/fe26bbe7-fdae-4d0a-98e6-3ac52c66f87f" />

```
df.set_index("PassengerId",inplace=True)
print(df.describe(),'\n',df.shape)
```

<img width="699" height="204" alt="image" src="https://github.com/user-attachments/assets/02925cd8-3b0d-4ebf-bc3f-82061549c5ac" />

```
df.nunique()
```

<img width="153" height="254" alt="image" src="https://github.com/user-attachments/assets/1991df8b-ce8f-419a-9385-9d317dca994f" />

```
sns.countplot(data=df,x='Survived')
df.Pclass.unique()
```

<img width="688" height="536" alt="image" src="https://github.com/user-attachments/assets/3ddc7156-fba3-4780-8191-1a8fecf35447" />

```
df.rename(columns={'Sex':'Gender'},inplace=True)
df
```

<img width="1110" height="442" alt="image" src="https://github.com/user-attachments/assets/47ad3611-bfca-4f5c-ba03-e1de30c7bfc0" />

```
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=7,aspect=.7)
```

<img width="752" height="549" alt="image" src="https://github.com/user-attachments/assets/44784116-224d-406c-ae74-00f5607b9ff9" />

```
df.boxplot(column="Age",by="Survived")
```

<img width="678" height="563" alt="image" src="https://github.com/user-attachments/assets/7fa974ef-f11c-4b07-8488-0de0aea8de60" />

```
sns.scatterplot(x=df["Age"],y=df["Fare"])
```

<img width="694" height="541" alt="image" src="https://github.com/user-attachments/assets/20874445-7795-4192-a9b1-cdcc54551206" />

```
fig,ax1=plt.subplots(figsize=(8,5))
plt = sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)
```

<img width="808" height="519" alt="image" src="https://github.com/user-attachments/assets/50a68f7d-53c4-4215-adbf-db55e19df2c2" />

```
plt = sns.boxplot(x='Pclass',y='Age',hue='Gender',data=df)
```

<img width="653" height="494" alt="image" src="https://github.com/user-attachments/assets/42e0195e-2ca8-42d9-895a-8955888f8f05" />

```
sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")
```

<img width="1147" height="572" alt="image" src="https://github.com/user-attachments/assets/d91c3102-bcc7-4f77-8f34-bf9d9996e7bc" />

```
sns.pairplot(data=df)
```

<img width="512" height="522" alt="image" src="https://github.com/user-attachments/assets/6b409366-50d9-4271-8674-c75227603d3c" />

```
corr1 = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr1, annot=True)
```

<img width="616" height="528" alt="image" src="https://github.com/user-attachments/assets/04c32e57-38d2-4eef-ab27-bba41aadb35f" />

## RESULT
Thus, Exploratory Data Analysis on the given data set is implemented successfully.

