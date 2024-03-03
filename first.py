import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Sample sprint data
sprint_data = {
    'sprint_number': 6,
    'velocity': [20, 18, 22, 19, 21, 20],
    'scope_volatility': ['low', 'medium', 'low', 'high', 'low', 'medium'],
    'commitment_reliability': ['high', 'medium', 'high', 'low', 'high', 'medium']
}

# NLG function to generate text summary
def generate_summary(sprint_data):
    sprint_number = sprint_data['sprint_number']
    velocity = sprint_data['velocity']
    scope_volatility = sprint_data['scope_volatility']
    commitment_reliability = sprint_data['commitment_reliability']

    # Generate text summary for sprint velocity
    velocity_change = velocity[-1] - velocity[-2]
    velocity_summary = f"In sprint {sprint_number}, the team achieved a velocity of {velocity[-1]} story points. "
    if velocity_change > 0:
        velocity_summary += f"This represents an increase of {velocity_change} story points compared to the previous sprint."
    elif velocity_change < 0:
        velocity_summary += f"This represents a decrease of {-velocity_change} story points compared to the previous sprint."
    else:
        velocity_summary += "Velocity remained unchanged compared to the previous sprint."

    # Generate text summary for scope volatility
    scope_volatility_summary = f"The scope volatility for sprint {sprint_number} was {scope_volatility[-1]}."

    # Generate text summary for commitment reliability
    commitment_reliability_summary = f"The commitment reliability for sprint {sprint_number} was {commitment_reliability[-1]}."

    # Combine all summaries
    summary = velocity_summary + '\n' + scope_volatility_summary + '\n' + commitment_reliability_summary

    return summary

# Generate and print text summary
summary = generate_summary(sprint_data)
print(summary)
