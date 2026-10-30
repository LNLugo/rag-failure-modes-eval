"""Generate corpus/ops500_fee_schedule_2025.pdf, a PDF whose key content is a table.

Tables in PDFs are the classic parsing failure: a plain text extractor flattens
them into rows of numbers, and fixed-size chunking then separates those rows
from the column headers that give the numbers their meaning.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[1] / "corpus" / "ops500_fee_schedule_2025.pdf"

HEADER = [
    "Account tier",
    "Monthly maintenance",
    "Domestic wire",
    "International wire",
    "Overdraft",
    "Stop payment",
]
ROWS = [
    ["Everyday Checking", "$12.00", "$30.00", "$45.00", "$35.00", "$30.00"],
    ["Everyday Plus", "$18.00", "$25.00", "$45.00", "$35.00", "$30.00"],
    ["Student Checking", "$0.00", "$30.00", "$50.00", "$15.00", "$30.00"],
    ["Senior Advantage", "$5.00", "$25.00", "$45.00", "$25.00", "$20.00"],
    ["Premier Checking", "$25.00", "$20.00", "$35.00", "$30.00", "$25.00"],
    ["Private Client", "$0.00", "$0.00", "$0.00", "$0.00", "$0.00"],
    ["Business Essentials", "$20.00", "$28.00", "$48.00", "$36.00", "$32.00"],
    ["Business Growth", "$40.00", "$22.00", "$42.00", "$36.00", "$32.00"],
    ["Business Premier", "$75.00", "$18.00", "$38.00", "$36.00", "$29.00"],
    ["Nonprofit Checking", "$8.00", "$16.00", "$44.00", "$34.00", "$27.00"],
]


def main() -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT), pagesize=landscape(letter), title="Fee Schedule 2025")
    story = [
        Paragraph("Harbor Point Bank Consumer and Business Fee Schedule 2025", styles["Title"]),
        Paragraph("Document ID: OPS-500. Effective date: 1 January 2025. Owner: Deposit Operations.", styles["Normal"]),
        Spacer(1, 8),
        Paragraph(
            "This schedule lists the standard fees for each deposit account tier offered by "
            "Harbor Point Bank. Fees are charged per item unless stated as monthly. Wire fees "
            "apply to outgoing wires; incoming wires are free for every tier. Fees may be "
            "waived at the discretion of a branch manager for customer service reasons.",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]
    table = Table([HEADER] + ROWS, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story += [
        table,
        Spacer(1, 12),
        Paragraph(
            "Monthly maintenance fees are waived for Everyday Checking when the average daily "
            "balance is $1,500 or more.",
            styles["Normal"],
        ),
    ]
    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
