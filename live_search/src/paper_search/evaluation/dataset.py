"""Ten fixed, diverse claims drawn from the existing query audit."""

from paper_search.evaluation.models import ClaimCase

CLAIMS = [
    ClaimCase(
        id="c01",
        domain="biomedical",
        claim=(
            "The results of this study show that the expression of TNF-α does "
            "not increase the risk of COVID-19 in older adults."
        ),
    ),
    ClaimCase(
        id="c02",
        domain="battery_engineering",
        claim=(
            "In this study, we demonstrate that lithium-ion batteries retain "
            "90% capacity after 500 cycles at -5 °C."
        ),
    ),
    ClaimCase(
        id="c03",
        domain="power_electronics",
        claim=(
            "Our experiments show that the efficiency of SiC MOSFETs is not "
            "lower than that of silicon devices at 100 kHz."
        ),
    ),
    ClaimCase(
        id="c04",
        domain="climate",
        claim=(
            'This study demonstrates that '
            'elevated atmospheric CO2 concentrations '
            "increase ocean acidification in coastal waters."
        ),
    ),
    ClaimCase(
        id="c05",
        domain="astronomy",
        claim=(
            "The presence of dark matter increases the rotational velocity "
            "of spiral galaxies."
        ),
    ),
    ClaimCase(
        id="c06",
        domain="agriculture",
        claim=(
            "Experimental results show that deficit irrigation reduces water "
            "use without decreasing wheat yield under drought conditions."
        ),
    ),
    ClaimCase(
        id="c07",
        domain="nutrition_uncertainty",
        claim=(
            "Preliminary findings suggest that vitamin D supplementation may "
            "reduce respiratory infections in older adults."
        ),
    ),
    ClaimCase(
        id="c08",
        domain="computer_science",
        claim=(
            "Our results show that graph neural networks improve molecular "
            "property prediction compared with random forests."
        ),
    ),
    ClaimCase(
        id="c09",
        domain="catalysis",
        claim=(
            "It has been demonstrated that platinum nanoparticles accelerate "
            "hydrogen evolution at 25 °C in acidic solutions."
        ),
    ),
    ClaimCase(
        id="c10",
        domain="general_science",
        claim="Exercise improves memory.",
    ),
]
