import re, string, calendar
from wikipedia import WikipediaPage
import wikipedia
from bs4 import BeautifulSoup
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from match import match
from typing import List, Callable, Tuple, Any, Match


def get_page_html(title: str) -> str:
    """Gets html of a wikipedia page

    Args:
        title - title of the page

    Returns:
        html of the page
    """
    results = wikipedia.search(title)
    return WikipediaPage(results[0]).html()


def get_first_infobox_text(html: str) -> str:
    """Gets first infobox html from a Wikipedia page (summary box)

    Args:
        html - the full html of the page

    Returns:
        html of just the first infobox
    """
    soup = BeautifulSoup(html, "html.parser")
    results = soup.find_all(class_="infobox")

    if not results:
        raise LookupError("Page has no infobox")
    return results[0].text


def clean_text(text: str) -> str:
    """Cleans given text removing non-ASCII characters and duplicate spaces & newlines

    Args:
        text - text to clean

    Returns:
        cleaned text
    """
    only_ascii = "".join([char if char in string.printable else " " for char in text])
    no_dup_spaces = re.sub(" +", " ", only_ascii)
    no_dup_newlines = re.sub("\n+", "\n", no_dup_spaces)
    return no_dup_newlines


def get_match(
    text: str,
    pattern: str,
    error_text: str = "Page doesn't appear to have the property you're expecting",
) -> Match:
    """Finds regex matches for a pattern

    Args:
        text - text to search within
        pattern - pattern to attempt to find within text
        error_text - text to display if pattern fails to match

    Returns:
        text that matches
    """
    p = re.compile(pattern, re.DOTALL | re.IGNORECASE)
    match = p.search(text)

    if not match:
        raise AttributeError(error_text)
    return match


def get_polar_radius(planet_name: str) -> str:
    """Gets the radius of the given planet

    Args:
        planet_name - name of the planet to get radius of

    Returns:
        radius of the given planet
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(planet_name)))
    pattern = r"(?:Polar radius.*?)(?: ?[\d]+ )?(?P<radius>[\d,.]+)(?:.*?)km"
    error_text = "Page infobox has no polar radius information"
    match = get_match(infobox_text, pattern, error_text)

    return match.group("radius")


def get_birth_date(name: str) -> str:
    """Gets birth date of the given person

    Args:
        name - name of the person

    Returns:
        birth date of the given person
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(name)))
    pattern = r"(?:Born\D*)(?P<birth>\d{4}-\d{2}-\d{2})"
    error_text = (
        "Page infobox has no birth information (at least none in xxxx-xx-xx format)"
    )
    match = get_match(infobox_text, pattern, error_text)

    return match.group("birth")

def get_university_motto(name: str) -> str:
    """Gets the motto(s) of the given university

    Args:
        name - name of the university

    Returns:
        motto(s) of the given university
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(name)))
    pattern = r"English(.*?)(?=Type)"
    error_text = (
        "Page infobox has no motto information"
    )
    match = get_match(infobox_text, pattern, error_text)
    return match.groups()[0]

def get_artist_genre(name: str) -> str:
    """Gets the genre(s) of the given musical artist

    Args:
        name - name of the musical artist

    Returns:
        genre(s) of the given musical artist
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(name)))
    pattern = r"Genres\n(.*?)(?=\n)"
    error_text = (
        "Page infobox has no genre information"
    )
    match = get_match(infobox_text, pattern, error_text)
    return match.groups()[0]

def get_painter_movement(painter_name: str) -> str:
    """Gets the movement of the given painter

    Args:
        painter_name - name of the painter to get movement of

    Returns:
        movement of the given painter
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(painter_name)))
    pattern = r"Movement\s*[:\-]?\s*(?P<movement>[^,]+?)(?=\s*(?:Family|Spouses|Known for|Notable work|Signature|Born|Died|Resting place|Education|Years active|Other names|Occupation|Works|$))"
    error_text = "Page infobox has no artistic movement information"
    match = re.search(pattern, infobox_text)
    return match.group("movement").strip() if match else error_text

def get_notable_work(painter_name: str) -> str:
    """Gets one notable work of the given painter

    Args:
        painter_name - name of the painter to get notable work of

    Returns:
        one notable work of the given painter
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(painter_name)))
    pattern = r"Notable work.*?([A-Za-z0-9\s,]+(?:\s*\(\d{4}(?:–\d{4})?\))?)\s*(?:painting|sculpture|artwork)?\s*(?:[^\n]*)"
    error_text = "Page infobox has no notable work information"
    match = get_match(infobox_text, pattern, error_text)
    return f"{match.group(1)}"

# below are a set of actions. Each takes a list argument and returns a list of answers
# according to the action and the argument. It is important that each function returns a
# list of the answer(s) and not just the answer itself.
def notable_work (matches: List[str]) -> List[str]:
    """Returns one notable work of named painter in matches

    Args:
        matches - match from pattern of painter's name to find one notable work of

    Returns:
        notable work of named painter
    """
    return [get_notable_work(" ".join(matches))]

def university_motto (matches: List[str]) -> List[str]:
    """Returns motto of named university in matches

    Args:
        matches - match from pattern of university's name to find motto(s) of

    Returns:
        motto of named university
    """
    return [get_university_motto(matches[0])]

def painter_movement (matches: List[str]) -> List[str]:
    """Returns artistic movement of named painter in matches

    Args:
        matches - match from pattern of painter's name to find artisitc movement of

    Returns:
        artistic movement of named painter
    """
    return [get_painter_movement(matches[0])]

def artist_genre(matches: List[str]) -> List[str]:
    """Returns the genre of named musical artist in matches

    Args:
        matches - match from pattern of person's name to find genre of

    Returns:
        genre of named person
    """
    return [get_artist_genre(matches[0])]


def birth_date(matches: List[str]) -> List[str]:
    """Returns birth date of named person in matches

    Args:
        matches - match from pattern of person's name to find birth date of

    Returns:
        birth date of named person
    """
    return [get_birth_date(" ".join(matches))]


def polar_radius(matches: List[str]) -> List[str]:
    """Returns polar radius of planet in matches

    Args:
        matches - match from pattern of planet to find polar radius of

    Returns:
        polar radius of planet
    """
    return [get_polar_radius(matches[0])]


# dummy argument is ignored and doesn't matter
def bye_action(dummy: List[str]) -> None:
    raise KeyboardInterrupt


# type aliases to make pa_list type more readable, could also have written:
# pa_list: List[Tuple[List[str], Callable[[List[str]], List[Any]]]] = [...]
Pattern = List[str]
Action = Callable[[List[str]], List[Any]]

# The pattern-action list for the natural language query system. It must be declared
# here, after all of the function definitions
pa_list: List[Tuple[Pattern, Action]] = [
    ("when was % born".split(), birth_date),
    ("what is the polar radius of %".split(), polar_radius),
    ("what is the movement of %".split(), painter_movement),
    ("what genre is %".split(), artist_genre),
    ("what is the motto of %".split(), university_motto),
    ("what is one work from %".split(), notable_work),
    (["bye"], bye_action),
]


def search_pa_list(src: List[str]) -> List[str]:
    """Takes source, finds matching pattern and calls corresponding action. If it finds
    a match but has no answers it returns ["No answers"]. If it finds no match it
    returns ["I don't understand"].

    Args:
        source - a phrase represented as a list of words (strings)

    Returns:
        a list of answers. Will be ["I don't understand"] if it finds no matches and
        ["No answers"] if it finds a match but no answers
    """
    for pat, act in pa_list:
        mat = match(pat, src)
        if mat is not None:
            answer = act(mat)
            return answer if answer else ["No answers"]

    return ["I don't understand"]


def query_loop() -> None:
    """The simple query loop. The try/except structure is to catch Ctrl-C or Ctrl-D
    characters and exit gracefully"""
    print("Welcome to the Wikipedia chatbot!\n")
    while True:
        try:
            print()
            query = input("Your query? ").replace("?", "").lower().split()
            answers = search_pa_list(query)
            for ans in answers:
                print(ans)

        except (KeyboardInterrupt, EOFError):
            break

    print("\nSo long!\n")


# uncomment the next line once you've implemented everything are ready to try it out
query_loop()

# Welcome to the Wikipedia chatbot!

# Your query? what is one work from Pablo Picasso
# La Vie (1903)

# Your query? what is one work from Van Gogh
# Sunflowers (1887)

# Your query? what is the movement of Picasso
# Cubism, Surrealism

# Your query? what is the movement of Leonardo da Vinci
# High Renaissance

# Your query? what is the motto of Northwestern University
# "Whatsoever things are true" (Philippians 4:8 AV)"The Word full of grace and truth" (John 1:14)

# Your query? what is the motto of Harvard University
# "Truth"

# Your query? what genre is Kendrick Lamar
# West Coast hip hop

# Your query? what genre is Rihanna
# R&B