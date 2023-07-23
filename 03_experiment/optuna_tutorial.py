import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def objective(trial):
    trial.suggest_int("n_estimators", 100, 1000, step=100)
    trial.suggest_int("max_depth", 3, 10)
    #
    # load data
    #
    iris = load_iris(as_frame=True)
    X, y = iris["data"], iris["target"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2024)

    #
    # train model
    #
    clf = RandomForestClassifier(
        n_estimators=trial.params["n_estimators"], max_depth=trial.params["max_depth"], random_state=2024
    )
    clf.fit(X_train, y_train)

    #
    # evaluate train model
    #
    y_pred = clf.predict(X_valid)
    acc_score = accuracy_score(y_valid, y_pred)
    return acc_score


if __name__ == "__main__":
    # study
    sampler = optuna.samplers.RandomSampler(seed=2024)
    study = optuna.create_study(sampler=sampler, study_name="hpo-tutorial", direction="maximize")

    # optimize
    study.optimize(objective, n_trials=5)
