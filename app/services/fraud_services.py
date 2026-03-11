from datetime import datetime, timedelta

def check_fraud(transaction, user_history):

    risk_score = 0


    if transaction.amount > 5000:
        risk_score += 40


    if user_history:
        avg_amount = sum(t.amount for t in user_history) / len(user_history)

        if transaction.amount > avg_amount * 3:
            risk_score += 40


    previous_locations = [t.location for t in user_history]

    if transaction.location not in previous_locations:
        risk_score += 20
    
    now=datetime.utcnow()
    recent_transaction = [t for t in user_history if t.created_at>=now - timedelta(seconds=120)]
    if len(recent_transaction) >= 5:
        risk_score += 50


    return risk_score >= 50