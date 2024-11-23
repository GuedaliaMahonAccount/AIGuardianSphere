# AIGuardianSphere


code colab drive: https://colab.research.google.com/drive/1YkhGfcsanss7DzrOB5ZfQJ3ivDua-xzX#scrollTo=R3dnqXSChSxi




we need to prepare a json with questions and answer like this: 
[
    {
        "question": "אני לא מרגיש טוב, מה לעשות?",
        "context": "",
        "answer": "אם זה כאב ראש, נסה כדור נגד כאבים. אם זה חמור, פנה לרופא כאן: [קישור]."
    },
    {
        "question": "איך אני מוצא רופא?",
        "context": "",
        "answer": "אתה יכול למצוא רופא בלינק הבא: [קישור]."
    }
]



and we will translate it in english with gpt:
[
    {
        "question": "I don't feel well, what should I do?",
        "context": "",
        "answer": "If it's a headache, try aspirin. If it's serious, consult a doctor here: [link]."
    },
    {
        "question": "How can I find a doctor?",
        "context": "",
        "answer": "You can find a doctor at the following link: [link]."
    }
]




and after that, we will train the model in the code
