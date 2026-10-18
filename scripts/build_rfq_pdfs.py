"""Render the fictitious customer RFQ / tender documents in rfq_pdf_content.py to PDF."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, KeepTogether, NextPageTemplate)
from reportlab.pdfgen import canvas
from xml.sax.saxutils import escape as X
from rfq_pdf_content import RFQS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "customer_rfq_pdfs")
os.makedirs(OUT, exist_ok=True)

NAVY = colors.HexColor("#22313F")
ACCENT = {"RFQ-2023-0105": colors.HexColor("#8A5A14"), "RFQ-2023-0094": colors.HexColor("#0B5563"),
          "RFQ-2026-0563": colors.HexColor("#2B4C7E")}
GREY = colors.HexColor("#6B6B6B")
LIGHT = colors.HexColor("#F1F1EE")

S = {
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13.5, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.5, leading=11),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=12.5, leading=16, spaceBefore=6, spaceAfter=8, textColor=NAVY),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=4, textColor=NAVY),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.8, leading=11.5),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8.8, leading=11.5, textColor=colors.white),
    "clauseno": ParagraphStyle("cn", fontName="Helvetica-Bold", fontSize=9.5, leading=13.5),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.5, leading=13.5, leftIndent=14, bulletIndent=3, spaceAfter=2),
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=17, leading=22, textColor=NAVY, spaceAfter=10),
    "doctype": ParagraphStyle("doctype", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=GREY, spaceAfter=6),
}


class NumberedCanvas(canvas.Canvas):
    """Adds 'Page x of y' after all pages are known."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        for st in self._saved:
            self.__dict__.update(st)
            self.setFont("Helvetica", 7.5)
            self.setFillColor(GREY)
            self.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {self._pageNumber} of {n}")
            super().showPage()
        super().save()


def make_doc(r, path):
    accent = ACCENT[r["rfq_id"]]

    def cover(c, d):
        c.saveState()
        c.setFillColor(accent)
        c.rect(0, A4[1] - 34 * mm, A4[0], 34 * mm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(18 * mm, A4[1] - 17 * mm, r["customer"])
        c.setFont("Helvetica", 8.5)
        c.drawString(18 * mm, A4[1] - 24 * mm, r["address"])
        c.drawString(18 * mm, A4[1] - 29 * mm, f"{r['division']}  |  {r['email']}")
        foot(c, d)
        c.restoreState()

    def later(c, d):
        c.saveState()
        c.setStrokeColor(accent)
        c.setLineWidth(1.2)
        c.line(18 * mm, A4[1] - 16 * mm, A4[0] - 18 * mm, A4[1] - 16 * mm)
        c.setFont("Helvetica-Bold", 8.5)
        c.setFillColor(accent)
        c.drawString(18 * mm, A4[1] - 13 * mm, r["customer"])
        c.setFont("Helvetica", 8.5)
        c.setFillColor(GREY)
        c.drawRightString(A4[0] - 18 * mm, A4[1] - 13 * mm, f"Enquiry No. {r['customer_ref']}")
        foot(c, d)
        c.restoreState()

    def foot(c, d):
        c.setStrokeColor(colors.HexColor("#CCCCCC"))
        c.setLineWidth(0.5)
        c.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(GREY)
        c.drawString(18 * mm, 10 * mm, "Confidential – issued for tendering purposes only. Not to be reproduced or disclosed.")
        c.setFont("Helvetica-Oblique", 6.5)
        c.drawString(18 * mm, 6.5 * mm, "Fictitious sample document generated for testing. Not a real tender; all names, addresses and references are invented.")

    doc = BaseDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=22 * mm, bottomMargin=20 * mm,
                          title=f"{r['customer_ref']} – {r['title']}", author=r["customer"], subject="Request for Quotation")
    fr_cover = Frame(18 * mm, 20 * mm, A4[0] - 36 * mm, A4[1] - 62 * mm, id="cover")
    fr = Frame(18 * mm, 20 * mm, A4[0] - 36 * mm, A4[1] - 42 * mm, id="body")
    doc.addPageTemplates([PageTemplate("cover", [fr_cover], onPage=cover), PageTemplate("later", [fr], onPage=later)])
    return doc


def table(rows, widths, accent, header=True, zebra=True):
    data = [[Paragraph(X(str(x)), S["cellb"] if (header and i == 0) else S["cell"]) for x in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), accent))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    return t


def short(t, n=75):
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0].rstrip(",;:") + " …"


def clause(cl):
    no, _cat, text, *_ = cl
    t = Table([[Paragraph(no, S["clauseno"]), Paragraph(X(text), S["body"])]], colWidths=[16 * mm, None])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def build(r):
    path = os.path.join(OUT, r["file"])
    doc = make_doc(r, path)
    a = ACCENT[r["rfq_id"]]
    W = A4[0] - 36 * mm
    s = [NextPageTemplate("later")]

    # cover
    s += [Spacer(1, 10 * mm), Paragraph("ENQUIRY / REQUEST FOR QUOTATION", S["doctype"]), Paragraph(X(r["title"]), S["title"]), Spacer(1, 4 * mm)]
    s.append(table([
        ("Enquiry No.", r["customer_ref"]), ("Date of issue", r["issue_date"]), ("Project", r["project"]),
        ("Last date for pre-bid queries", r["query_deadline"]), ("Bid due date and time", r["due"]),
        ("Mode of submission", r["submission"]), ("Bid validity", r["validity"]), ("Currency of offer", r["currency"]),
        ("Contact", f"{r['contact']}, {r['email']}"),
    ], [55 * mm, W - 55 * mm], a, header=False))
    s += [Spacer(1, 8 * mm), Paragraph("Contents", S["h2"])]
    toc = ["1. Instructions to bidders", "2. Scope of supply", "3. Technical specification", "4. Approved makes",
           "5. Inspection and testing", "6. Documentation", "7. Commercial terms and conditions",
           "Annexure A – Price schedule", "Annexure B – Technical and commercial compliance statement", "Annexure C – Required delivery schedule"]
    s += [Paragraph(x, S["small"]) for x in toc]
    s.append(PageBreak())

    # 1
    s += [Paragraph("1. Instructions to bidders", S["h1"]), Paragraph(X(r["intro"]), S["body"])]
    s += [Paragraph("1.1 Bidders shall submit the following with their offer:", S["body"])]
    for b in ["Technical offer with a description of the proposed design, preliminary general arrangement drawing and data sheets.",
              "Completed price schedule (Annexure A). The technical offer shall not contain prices.",
              "Completed compliance statement (Annexure B), listing every deviation clause by clause. Deviations not listed will be deemed not taken.",
              "Delivery schedule against the milestones in Annexure C.",
              "Reference list of similar equipment supplied in the last five (5) years, with end-user contacts.",
              "Quality plan and inspection and test plan (ITP)."]:
        s.append(Paragraph(b, S["bullet"], bulletText="•"))
    s.append(Paragraph(f"1.2 Queries shall be raised in writing to {r['email']} not later than {r['query_deadline']}. Replies will be shared with all bidders.", S["body"]))
    s.append(Paragraph("1.3 Incomplete offers, or offers received after the due date and time, may be rejected.", S["body"]))

    # 2
    s += [Paragraph("2. Scope of supply", S["h1"]), Paragraph("2.1 The bidder's scope includes:", S["body"])]
    s += [Paragraph(X(x), S["bullet"], bulletText="•") for x in r["scope"]]
    s += [Paragraph("2.2 The following are excluded from the bidder's scope and will be provided by the Purchaser:", S["body"])]
    s += [Paragraph(X(x), S["bullet"], bulletText="•") for x in r["exclusions"]]

    # 3
    s += [PageBreak(), Paragraph("3. Technical specification", S["h1"]), Paragraph("3.1 Design data", S["h2"])]
    s.append(table(r["design_table"], [58 * mm, W - 58 * mm], a))
    for head, cls in r["clauses"].items():
        s.append(Paragraph(head, S["h2"]))
        s += [clause(c) for c in cls]

    # 4
    s += [Paragraph("4. Approved makes", S["h1"]), Paragraph(X(r["makes_intro"]), S["body"]), clause(r["makes_clause"]), Spacer(1, 3),
          KeepTogether([Paragraph("Table 4.1", S["small"]), table(r["makes"], [60 * mm, W - 60 * mm], a)])]

    # 5, 6
    s += [Paragraph("5. Inspection and testing", S["h1"])] + [clause(c) for c in r["testing"]]
    s += [Paragraph("6. Documentation", S["h1"])] + [clause(c) for c in r["docs"]]

    # 7
    s += [Paragraph("7. Commercial terms and conditions", S["h1"])] + [clause(c) for c in r["commercial"]]
    nxt = f"7.{len(r['commercial']) + 1}"
    s.append(clause((nxt, "", "In case of conflict, this enquiry shall prevail over the Purchaser's general conditions of contract and standard terms of purchase.")))

    # Annexures
    s += [PageBreak(), Paragraph("Annexure A – Price schedule", S["h1"]),
          Paragraph(f"All prices in {r['currency']}. Bidders shall not alter the format. Add rows for options if required.", S["body"])]
    rows = [("S. No.", "Description", "Qty", f"Unit price ({r['currency']})", f"Total ({r['currency']})", "Remarks")]
    rows += [(str(i), d, "", "", "", "") for i, d in enumerate(r["price_rows"], 1)]
    rows += [("", "Grand total", "", "", "", "")]
    s.append(table(rows, [13 * mm, 70 * mm, 13 * mm, 28 * mm, 25 * mm, W - 149 * mm], a, zebra=False))

    s += [Spacer(1, 8 * mm), Paragraph("Annexure B – Technical and commercial compliance statement", S["h1"]),
          Paragraph("Bidders shall state compliance against every clause of Sections 3 to 7. Use 'C' for comply, 'D' for deviation. "
                    "Clarifications and deviations shall be described in full.", S["body"])]
    allc = [c for cls in r["clauses"].values() for c in cls] + [r["makes_clause"]] + r["testing"] + r["docs"] + r["commercial"]
    rows = [("Clause", "Requirement (short)", "C / D", "Bidder's deviation or clarification")]
    rows += [(c[0], short(c[2]), "", "") for c in allc]
    s.append(table(rows, [16 * mm, 88 * mm, 14 * mm, W - 118 * mm], a, zebra=False))

    s += [Spacer(1, 8 * mm), KeepTogether([Paragraph("Annexure C – Required delivery schedule", S["h1"]),
                                           table(r["delivery_table"], [100 * mm, W - 100 * mm], a)])]
    s += [Spacer(1, 12 * mm), KeepTogether([
        Paragraph(f"For and on behalf of {r['customer']}", S["body"]), Spacer(1, 10 * mm),
        Paragraph(f"{r['contact']}<br/>{r['division']}", S["small"])])]

    doc.build(s, canvasmaker=NumberedCanvas)
    return path, allc


if __name__ == "__main__":
    for r in RFQS:
        p, _ = build(r)
        print(p)
