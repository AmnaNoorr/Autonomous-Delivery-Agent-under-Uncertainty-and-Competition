# config.py
from oel_instance import make_instance, stamp

MEMBERS = [
    ("Amna Noor", "486679"),
    ("Fatima Aleem", "460798"),
    ("Laraib Zahra", "486986")
]

cms_ids = [cms for _, cms in MEMBERS]
inst, grid = make_instance(cms_ids)

if __name__ == "__main__":
    print(stamp(MEMBERS))
    print("Instance ID:", inst["instance_id"])
    print("Twists:", inst["twists"])