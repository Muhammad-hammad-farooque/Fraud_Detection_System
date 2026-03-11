def calculate_risk(transaction, user_transactions):
    risk_score=0


    if transaction.amount > 5000:
        risk_score+=0.4

    if user_transactions:
        avg_amount=sum(t.amount for t in user_transactions)/len(user_transactions)
        if transaction.amount>avg_amount*3:
            risk_score+=0.4

    locations = [t.location for t in user_transactions]

    if transaction.location not in locations:
        risk_score+=0.2

    return risk_score