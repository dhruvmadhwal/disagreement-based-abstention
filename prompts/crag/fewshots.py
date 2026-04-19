from typing import List, Dict, Any


def open_ended_examples() -> List[Dict[str, str]]:
    return [
        {"q": "When was the director of The Shape of Water born?", "answer": "1964-10-09"},
        {"q": "How many children does the lead actor of Pirates of the Caribbean: The Curse of the Black Pearl have?", "answer": "2"},
        {"q": "Who was the quarterback of the team that won Super Bowl XLI?", "answer": "Peyton Manning"},
        {"q": "In what country was the singer of \"Rehab\" born?", "answer": "United Kingdom"},
        {"q": "What is the capital of the country where the Colosseum is located?", "answer": "Rome"},
        {"q": "What is the highest mountain in the country that uses the yen?", "answer": "Mount Fuji"},
        {"q": "What party did the U.S. president during the Cuban Missile Crisis belong to?", "answer": "Democratic Party"},
        {"q": "Which is older: the founder of SpaceX or the founder of Amazon?", "answer": "Jeff Bezos"},
        {"q": "Name two official languages of the country whose capital is Bern.", "answer": "German; French"},
        {"q": "Which river flows through the capital of the country whose flag features a red maple leaf?", "answer": "Ottawa River"},
        {"q": "Where was the director of Roma born?", "answer": "Mexico City"},
        {"q": "Who was the point guard of the team that won the NBA Finals in 2015?", "answer": "Stephen Curry"},
    ]


def assistive_examples() -> List[Dict[str, Any]]:
    return [
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Who directed the film The Shape of Water?")',
                'answer_2: str = qa_model("When was {answer_1} born?", answer_1=answer_1)',
            ],
            "answers": {"answer_1": "Guillermo del Toro", "answer_2": "1964-10-09"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Who was the lead actor of Pirates of the Caribbean: The Curse of the Black Pearl?")',
                'answer_2: str = qa_model("How many children does {answer_1} have?", answer_1=answer_1)',
            ],
            "answers": {"answer_1": "Johnny Depp", "answer_2": "2"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Who is the CEO of Microsoft?")',
                'answer_2: str = qa_model("How old is {answer_1}?", answer_1=answer_1)',
            ],
            "answers": {"answer_1": "Satya Nadella", "answer_2": "55"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Which country hosted the first FIFA World Cup?")',
                'answer_2: str = qa_model("Which country hosted the second FIFA World Cup?")',
                'answer_3: str = qa_model("Which country hosted the third FIFA World Cup?")',
                'answer_4: str = qa_model("What are the capital cities of {answer_1}, {answer_2} and {answer_3}", answer_1=answer_1, answer_2=answer_2, answer_3=answer_3)',
            ],
            "answers": {"answer_1": "Uruguay", "answer_2": "Italy", "answer_3": "France", "answer_4": "Montevideo; Rome; Paris"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Which team won Super Bowl XLI?")',
                'answer_2: str = qa_model("Who was the quarterback for {answer_1} in Super Bowl XLI?", answer_1=answer_1)',
            ],
            "answers": {"answer_1": "Indianapolis Colts", "answer_2": "Peyton Manning"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Who was the U.S. president during the Cuban Missile Crisis?")',
                'answer_2: str = qa_model("What party did {answer_1} belong to?", answer_1=answer_1)',
            ],
            "answers": {"answer_1": "John F. Kennedy", "answer_2": "Democratic Party"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Which country uses the yen?")',
                'answer_2: str = qa_model("What is the highest mountain in {answer_1}?", answer_1=answer_1)',
            ],
            "answers": {"answer_1": "Japan", "answer_2": "Mount Fuji"},
        },
        {
            "dsl_lines": [
                'answer_1: str = qa_model("Whar are the two official languages of the country whose capital is Bern?")',
            ],
            "answers": {"answer_1": "German; French"},
        },
    ]


