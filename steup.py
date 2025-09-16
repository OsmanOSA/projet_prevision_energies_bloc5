from setuptools import find_packages, setup
from typing import List, Literal

def get_requirements() -> List[str]:
    
    requirement_lst: List[str] = []
    try: 
        with open("requirements.txt", "r") as file:

            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement!="-e .":
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("Fichier requirements.txt non trouvé")

    return requirement_lst


setup(
    name="Prévisions énergétiques", 
    version="0.0.1", 
    author="Osman SAID ALI",
    author_email="saidaliosman925@gmai.com",
    packages=find_packages(),
    install_requires=get_requirements()

)


