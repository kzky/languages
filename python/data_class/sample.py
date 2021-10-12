from dataclasses import dataclass, asdict, astuple
from typing import Any, List


@dataclass
class Person:
    first_name: str
    last_name: str
    age: int
    birth_day: str  # want to add the format check
    address: str
    desc: Any = "no explanation!"


@dataclass
class Room:
    name: str
    in_person: List[Person]
    location: Any = None
    current_num_of_person: int = None
    total_num_of_person: int = None
    

def main():
    # Example 1
    person = Person(first_name="Kazuki", last_name="Yoshiyama",
                    age=32, birth_day="19870412", address="Stitzenburg str.")
    print(person)
    print(asdict(person))
    print(astuple(person))

    # Example 2
    person0 = Person(first_name="Kazuki", last_name="Yoshiyama",
                    age=32, birth_day="19870412", address="Stitzenburg str.")
    person1 = Person(first_name="Natsuha", last_name="Yoshiyama",
                    age=3, birth_day="20170810", address="Stitzenburg str.")
    room = Room(name="Room 84", in_person=[person0, person1])
    print(room)
    

if __name__ == '__main__':
    main()
