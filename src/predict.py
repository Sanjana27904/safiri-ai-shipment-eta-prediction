import pandas as pd
import joblib


# Load trained model
model = joblib.load(
    "../data/eta_prediction_model.pkl"
)


def predict_eta(
    departure_delay_hours,
    scheduled_transit_hours,
    estimated_arrival,
    scheduled_arrival,
    ship,
    dep_port,
    arr_port,
    route,
    scheduled_departure
):

    # Convert timestamps to datetime
    scheduled_arrival = pd.to_datetime(
        scheduled_arrival
    )

    scheduled_departure = pd.to_datetime(
        scheduled_departure
    )

    estimated_arrival = pd.to_datetime(
        estimated_arrival
    )

    # Calculate ETA update gap
    eta_update_gap = (
        estimated_arrival -
        scheduled_arrival
    ).total_seconds() / 3600

    # Prepare input features
    features = pd.DataFrame([{

        "departure_delay_hours":
            departure_delay_hours,

        "scheduled_transit_hours":
            scheduled_transit_hours,

        "eta_update_gap_hours":
            eta_update_gap,

        "scheduled_departure_hour":
            scheduled_departure.hour,

        "scheduled_departure_dayofweek":
            scheduled_departure.dayofweek,

        "ship":
            ship,

        "depPort":
            dep_port,

        "arrPort":
            arr_port,

        "route":
            route
    }])

    # Predict arrival delay
    predicted_delay = model.predict(
        features
    )[0]

    # Calculate predicted ETA
    predicted_eta = (
        scheduled_arrival +
        pd.Timedelta(
            hours=predicted_delay
        )
    )

    return predicted_delay, predicted_eta


# Example prediction
delay, eta = predict_eta(

    departure_delay_hours=2.0,

    scheduled_transit_hours=48.0,

    estimated_arrival="2026-09-08 18:00",

    scheduled_arrival="2026-09-08 16:00",

    ship="Megastar",

    dep_port="EETLL",

    arr_port="FIHEL",

    route="EETLL → FIHEL",

    scheduled_departure="2026-09-06 16:00"
)


print(
    f"Predicted delay: {delay:.2f} hours"
)

print(
    f"Predicted ETA: {eta.strftime('%Y-%m-%d %H:%M')}"
)