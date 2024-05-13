import os

from dotenv import load_dotenv
import asyncio

from utils import (
    fetch_rating_and_review,
    fetch_unique_values_from_db,
    generate_summary_openai,
    preprocess_reviews,
    update_review_summary,
)

load_dotenv()


async def main():
    review_table = os.getenv("REVIEW_TABLE")
    summary_table = os.getenv("SUMMARY_TABLE")

    # 1. get unique ids from review table
    unique_ids = fetch_unique_values_from_db(review_table, "Id")
    print("unique_ids : ", unique_ids)

    for _id in unique_ids:
        try:
            print("Generating summary for id : ", _id)

            # Fetch rating and review based on Id
            rating_review_df = fetch_rating_and_review(review_table, _id)

            # Calculate total rating
            total_rating = len(rating_review_df["Rating"])

            # Calculate average rating
            average_rating = round(rating_review_df["Rating"].mean(), 2)

            # Get all reviews in a list
            all_reviews = rating_review_df["Review"].to_list()

            # Prereprocess reviews
            all_reviews = preprocess_reviews(all_reviews)

            # Generate Summary
            summary = await generate_summary_openai(
                all_reviews, total_rating, average_rating
            )

            print("Summary : \n", summary)

            # Save summary to summary table
            update_review_summary(summary_table, _id, summary)

            print("\nFinished generating summary for id : ", _id)

            print("=" * 80 + "\n")
        except Exception as e:
            print("Error : ", e)


if __name__ == "__main__":
    asyncio.run(main())
