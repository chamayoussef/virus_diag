from sklearn.ensemble import RandomForestClassifier

def create_model():
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    return model