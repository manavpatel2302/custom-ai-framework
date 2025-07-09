# parts_recommender.py

def recommend_parts(availability_score):
    if availability_score >= 0.8:
        return "Use Supplier A - High Availability Parts"
    elif availability_score >= 0.5:
        return "Use Supplier B - Moderate Availability Parts"
    else:
        return "Use Supplier C - Low Availability Parts / Expect Delays"

if __name__ == "__main__":
    # Example usage with a chosen availability score:
    availability_score = 0.7
    recommendation = recommend_parts(availability_score)
    print(f"Parts Recommendation for availability score {availability_score:.2f}: {recommendation}")
