from typing import Any, Dict, List


def open_ended_examples() -> List[Dict[str, str]]:
    return [
        {"q": "Which company acquired Instagram and when was the CEO of that company born?", "answer": "1984-05-14"},
        {"q": "What city is the birthplace of the founder of SpaceX?", "answer": "Pretoria"},
        {"q": "Who wrote the novel that inspired the film Blade Runner?", "answer": "Philip K. Dick"},
        {"q": "Name the capital of the country where Machu Picchu is located.", "answer": "Lima"},
        {"q": "Who is older: the creator of Star Wars or the director of Titanic?", "answer": "George Lucas"},
        {"q": "Which river flows through the city that hosts the Eiffel Tower?", "answer": "Seine"},
        {"q": "Name two languages spoken in Switzerland.", "answer": "German; French"},
        {"q": "What is the home stadium of the NFL team owned by Jerry Jones?", "answer": "AT&T Stadium"},
        {"q": "Where was the author of Pride and Prejudice born?", "answer": "Steventon"},
        {"q": "Who composed the music for the film Inception?", "answer": "Hans Zimmer"},
        {"q": "What is the capital of the country whose flag features a maple leaf?", "answer": "Ottawa"},
        {"q": "Which planet is known as the Red Planet?", "answer": "Mars"},
    ]


def assistive_examples() -> List[Dict[str, Any]]:
    return [
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Who founded SpaceX?")',
                'answer_2: str = qa_model("Where was {answer_1} born?", answer_1=answer_1)',
            ],
            "answers": {"answer_1": "Elon Musk", "answer_2": "Pretoria"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Which team plays home games at AT&T Stadium?")',
                'answer_2: str = qa_model("Who owns {answer_1}?", answer_1=answer_1)',
            ],
            "answers": {"answer_1": "Dallas Cowboys", "answer_2": "Jerry Jones"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Who directed Titanic?")',
                'answer_2: str = qa_model("Who created Star Wars?")',
                'answer_3: str = qa_model("Who is older, {answer_1} or {answer_2}?", answer_1=answer_1, answer_2=answer_2)',
            ],
            "answers": {"answer_1": "James Cameron", "answer_2": "George Lucas", "answer_3": "George Lucas"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Which novel inspired the film Blade Runner?")',
                'answer_2: str = qa_model("Who wrote {answer_1}?", answer_1=answer_1)',
            ],
            "answers": {"answer_1": "Do Androids Dream of Electric Sheep?", "answer_2": "Philip K. Dick"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Which country is Machu Picchu located in?")',
                'answer_2: str = qa_model("What is the capital of {answer_1}?", answer_1=answer_1)',
            ],
            "answers": {"answer_1": "Peru", "answer_2": "Lima"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Which river runs through Paris?")',
                'answer_2: str = qa_model("What famous landmark is located in Paris?")',
            ],
            "answers": {"answer_1": "Seine", "answer_2": "Eiffel Tower"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Who composed the score for Inception?")',
                'answer_2: str = qa_model("What nationality is {answer_1}?", answer_1=answer_1)',
            ],
            "answers": {"answer_1": "Hans Zimmer", "answer_2": "German"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Which official languages are spoken in Switzerland?")',
            ],
            "answers": {"answer_1": "German; French; Italian; Romansh"},
        },
    ]

