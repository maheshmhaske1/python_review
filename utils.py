import os

import mariadb
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def fetch_unique_values_from_db(table, column_name):
    """
    Fetches unique values from a specified column in a given table in the MariaDB database.

    Args:
        table (str): The name of the table from which to fetch unique values.
        column_name (str): The name of the column from which to fetch unique values.

    Returns:
        list: A list of unique values from the specified column in the specified table.
              Returns an empty list if an error occurs during the database operations.

    Raises:
        None

    Notes:
        - Establishes a connection to the MariaDB database using the environment variables
          MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, and MYSQL_DB.
        - Creates a cursor object using the connection.
        - Executes an SQL query to select distinct values from the specified column in the specified table.
        - Fetches all the unique values from the query result.
        - Closes the cursor and connection properly in the finally block.

    Example:
        >>> fetch_unique_values_from_db("users", "email")
        ['user1@example.com', 'user2@example.com', 'user3@example.com']
    """
    conn = None
    curr = None
    try:
        # Establish a connection to the MariaDB database
        conn = mariadb.connect(
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            host=os.getenv("MYSQL_HOST"),
            database=os.getenv("MYSQL_DB"),
        )

        # Create a cursor object using the connection
        curr = conn.cursor()

        # SQL query to select unique IDs from a table
        curr.execute(f"SELECT DISTINCT {column_name} FROM {table}")

        # Fetch all unique IDs from the query
        unique_ids = [row[0] for row in curr.fetchall()]

        # Return the list of unique IDs
        return unique_ids

    except mariadb.Error as e:
        # Handle any errors that occur during the database operations
        print(f"Error: {e}")
        return []  # Return an empty list in case of error

    finally:
        # Ensure the cursor and connection are closed properly
        if curr:
            curr.close()
        if conn:
            conn.close()


def fetch_rating_and_review(table, id):
    """
    Fetches the rating and review for a specific ID from a given table in the MariaDB database.

    Parameters:
        table (str): The name of the table from which to fetch the rating and review.
        id (int): The ID of the rating and review to fetch.

    Returns:
        pandas.DataFrame or None: A DataFrame containing the rating and review if found, None otherwise.

    Raises:
        mariadb.Error: If an error occurs during the database operations.

    Notes:
        - Establishes a connection to the MariaDB database using the environment variables
          MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, and MYSQL_DB.
        - Creates a cursor object using the connection.
        - Executes an SQL query to select the rating and review based on the given ID.
        - Fetches the result of the query.
        - If a result is found, converts it to a DataFrame and returns it.
        - If no result is found, returns None.
        - Closes the cursor and connection properly in the finally block.

    Example:
        >>> fetch_rating_and_review("reviews", 123)
             Rating                                             Review
        0       4.5  Lorem ipsum dolor sit amet, consectetur adipiscing...
    """
    try:
        # Establish a connection to the MariaDB database
        conn = mariadb.connect(
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            host=os.getenv("MYSQL_HOST"),
            database=os.getenv("MYSQL_DB"),
        )
        # Create a cursor object using the connection
        curr = conn.cursor()

        # SQL query to select rating and review based on Id
        curr.execute(f"SELECT Rating, Review FROM {table} WHERE Id=?", (id,))

        # Fetch the result
        result = curr.fetchall()  # This gets the first row of the results

        if result:
            # Convert to DataFrame
            df = pd.DataFrame(result, columns=["Rating", "Review"])
            return df
        else:
            return None

    except mariadb.Error as e:
        print(f"Error: {e}")
        return None

    finally:
        if curr:
            curr.close()
        if conn:
            conn.close()


def preprocess_reviews(all_reviews):
    """
    Preprocesses a list of reviews by removing None values, empty strings, and duplicate reviews.

    Args:
        all_reviews (List[str]): A list of reviews to be preprocessed.

    Returns:
        List[str]: A list of preprocessed reviews with None values removed, empty strings removed, and duplicates removed.
    """
    # Remove None
    all_reviews = list(filter(None, all_reviews))
    # remove empty text
    all_reviews = [review for review in all_reviews if review]
    # Strip extra spaces
    all_reviews = [review.strip() for review in all_reviews]
    # Remove duplicates
    all_reviews = list(set(all_reviews))

    return all_reviews


def num_tokens_from_string(string: str, encoding=encoding) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens


async def generate_summary_openai(all_reviews, total_rating, average_rating):
    """
    Generates a summary of Google reviews for a sports academy using OpenAI's GPT-3.5-turbo model.

    Args:
        all_reviews (List[str]): A list of Google reviews for the sports academy.
        total_rating (int): The total rating of the sports academy.
        average_rating (float): The average rating of the sports academy.

    Returns:
        str: The generated summary of the Google reviews.

    Raises:
        None

    Notes:
        - The generated summary is between 350 and 500 words long and includes bullet points.
        - The tone of the summary is professional and includes facts from the Google reviews.
        - The summary is optimized for SEO and includes keywords related to sports academies.
        - The system prompt is used as a template for generating the summary.
        - The user prompt includes the reviews, total rating, and average rating.
        - The completion is generated using OpenAI's GPT-3.5-turbo model.
        - The summary is extracted from the completion's message content.

    Example:
        >>> reviews = ["Great facility with top-notch equipment.", "Excellent staff and friendly environment."]
        >>> total_rating = 45
        >>> average_rating = 4.5
        >>> generate_summary_openai(reviews, total_rating, average_rating)
    """

    # Create system prompt
    system_prompt_oldv1 = """Summarise these google reviews for a sports Academy in minimum 350 words and maximum 500 words with bullet points. Do not add pretext, directly start with summary. Make sure tone is professional with facts from google reviews. This is for marketing purpose and use SEO optimised keywords related to sports academies to improve search rankings."""
    system_prompt_oldv2 = """Please provide a summary of the Google reviews for a sports academy. The summary should be between 350 and 500 words and formatted with bullet points. Start directly with the summary without any introduction. Ensure the tone is professional, focusing on factual information derived from the reviews. This content is intended for marketing purposes, so incorporate SEO-optimized keywords related to sports academies to enhance search rankings. Additionally, ensure that each sentence ends with a full stop to maintain proper punctuation."""
    
    system_prompt = """Generate a detailed long summary of the Google reviews for a sports academy, ensuring the summary contains at least 350 words and does not exceed 500 words. Do not generate short summaries. Format the summary with bullet points. Start immediately with the summary, maintaining a professional tone and focusing on factual content from the reviews. This is for marketing purposes,so write summary as if you are promoting the sports academy. Include SEO-optimized keywords relevant to sports academies to boost search rankings. Make sure to include proper punctuation, especially full stops at the end of each sentence. Never generate less than 350 words."""
    max_tokens = 1200
    system_prompt_token_count = num_tokens_from_string(system_prompt)

    max_token_count_for_context = 16000 - max_tokens - system_prompt_token_count

    context = ""

    for _review in all_reviews:
        if num_tokens_from_string(_review) > max_token_count_for_context:
            break
        context += _review
        context += "\n"

    user_prompt = f"""Reviews:\nTotal Rating:{total_rating}\nAverage rating:{average_rating}\n{context}\n\nSummary:"""

    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
    )

    summary = completion.choices[0].message.content

    return summary


def update_review_summary(summary_table, id_to_update, new_review_summary):
    """
    A function to update the review summary in a specified table for a given ID with a new review summary.

    Parameters:
        summary_table (str): The name of the table where the review summary is to be updated.
        id_to_update (int): The ID for which the review summary is to be updated.
        new_review_summary (str): The new review summary to replace the existing summary.

    Returns:
        None
    """
    try:
        # Establish a connection to the MariaDB database
        conn = mariadb.connect(
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            host=os.getenv("MYSQL_HOST"),
            database=os.getenv("MYSQL_DB"),
        )
        # Create a cursor object using the connection
        curr = conn.cursor()

        # SQL query to update review_summary based on Id
        #curr.execute(
        #    f"UPDATE {summary_table} SET review_summary=? WHERE id=?;",
        #    (new_review_summary, id_to_update,),
        #)

        curr.execute("UPDATE bmp_academy_details SET review_summary = ? WHERE id = ?", (new_review_summary, id_to_update))

        # Commit the changes to the database
        conn.commit()

        print("\n Summary added in DB successfully.")

    except mariadb.Error as e:
        print(f"Error: {e}")

    finally:
        if curr:
            curr.close()
        if conn:
            conn.close()
