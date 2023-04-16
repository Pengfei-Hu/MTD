import Levenshtein
import os
import cv2
import pdfplumber
import numpy as np
import fitz
import glob
import re
from collections import Counter

LATEX_GREEK_MAP = {
    r'\alpha':'α', r'\Beta':'B', r'\delta':'δ', r'\Epsilon':'E', r'\Zeta':'Z', r'\theta':'θ', r'\iota':'ι', r'\Kappa':'K', r'\mu':'μ', 
    r'\Alpha':'A', r'\gamma':'γ', r'\Delta':'Δ', r'\varepsilon':'ε', r'\eta':'η', r'\Theta':'Θ', r'\Iota':'I', r'\lambda':'λ', r'\Mu':'M', 
    r'\beta':'β', r'\Gamma':'Γ', r'\epsilon':'ϵ', r'\zeta':'ζ', r'\Eta':'H', r'\vartheta':'ϑ', r'\kappa':'κ', r'\Lambda':'Λ', r'\nu':'ν', 
    r'\Nu':'N', r'\xi':'ξ', r'\Xi':'Ξ', r'\omicron':'ο', r'\Sigma':'Σ', r'\Tau':'T', r'\phi':'ϕ', r'\chi':'χ', r'\Chi':'X', r'\psi':'ψ',
    r'\Omicron':'O', r'\pi':'π', r'\Pi':'Π', r'\varpi':'ϖ', r'\varsigma':'ς', r'\upsilon':'υ', r'\Phi':'Φ', r'\Psi':'Ψ', r'	\Omega':'Ω',
    r'\rho':'ρ', r'\Rho':'P', r'\varrho':'ϱ', r'\sigma':'σ', r'\tau':'τ', r'\Upsilon':'Υ', r'\varphi':'φ', r'\omega':'ω', r'\\prime':'′',
}

REMOVE_LATEX_COMMAND = {
    'ref':r"\\ref{.*?}",
    'rm':r"\\rm ",
}

BRACE_LATEX_COMMAND = r"\\\w*?\*?\{(.*?)\}"
TEXORPDFSTRING = r""
BRACE_LEFT_COMMAND = r"\\\w*?\*?\{" 


def match_right_brace(input_str):
    assert input_str.startswith('{')
    num_left_brace, num_right_brace = 0, 0
    for c_i, c in enumerate(input_str):
        if c == '{':
            num_left_brace += 1
        if c == '}':
            num_right_brace += 1
        if num_left_brace == num_right_brace:
            break
    
    return input_str[:c_i + 1], c_i


def norm_latex_repr(latex_str):
    repr_latex = latex_str

    for k, v in REMOVE_LATEX_COMMAND.items():
        to_remove_latex_lst = re.findall(v, latex_str)
        for remove_str in to_remove_latex_lst:
            repr_latex = repr_latex.replace(remove_str, '')

    match_str = r'\texorpdfstring{'
    if match_str in repr_latex:
        start_idx_lst = [each.start() for each in re.finditer(r'\\texorpdfstring{', repr_latex)]
        texorpdf_brace_1_lst = []
        for s_i in start_idx_lst:
            texorpdf_brace_1, end_idx_1 = match_right_brace(repr_latex[s_i + len(match_str) - 1:])
            texorpdf_brace_1_lst.append(texorpdf_brace_1)
        for texorpdf_brace_1 in texorpdf_brace_1_lst:
            repr_latex = repr_latex.replace(texorpdf_brace_1, '')
        repr_latex = repr_latex.replace(r'\texorpdfstring', '')


    to_remove_latex_lst = re.findall(BRACE_LEFT_COMMAND, repr_latex)
    for to_remove in to_remove_latex_lst:
        repr_latex = repr_latex.replace(to_remove, '')

    repr_latex = repr_latex.replace('\\', '')

    repr_latex = repr_latex.replace('{', '').replace('}', '').replace('^', '').replace('_', '')

    return repr_latex



def repr_latex(latex_str):
    latex_str = latex_str.replace('~', '')
    if len(latex_str.replace('{', '')) != len(latex_str.replace('}', '')):
        if not len(latex_str.replace('{', '')) + 1 == len(latex_str.replace('}', '')):
            pass
        else:
            latex_str = latex_str + '}'
    math_latex_lst = re.findall(r"\$.*?\$", latex_str)
    latex_repr = latex_str
    for math_latex in math_latex_lst:
        repr_math = math_latex
        for k, v in LATEX_GREEK_MAP.items():
            repr_math = repr_math.replace(k, v)
        repr_math = norm_latex_repr(repr_math)
        repr_math = repr_math.replace('$', '')
        latex_repr = latex_repr.replace(math_latex, repr_math)
    latex_repr = norm_latex_repr(latex_repr)   
    return latex_repr


def repr_pdf(pdf_content):
    pdf_content = pdf_content.strip()
    if pdf_content.lower().startswith('appendix'):
        pdf_content = pdf_content[len('appendix'):]
        pdf_content = pdf_content.strip()
    match_with_blank = r"[a-zA-Z][\.\:]? "
    match = re.match(match_with_blank, pdf_content)
    if match:
        group = match.group(0)
        pdf_content = pdf_content.replace(group, '')
    repr_pdf = pdf_content.replace(' ', '')
    remove_re_lst = '|'.join([r"(\d+[\.\:]?)+", r"(XC|XL|L?X{0,3})(IX|XI*V?I*X?|IX|IV|V?I{0,3})[\.\:]?"])
    match = re.match(remove_re_lst, repr_pdf)
    if match:
        group = match.group(0)
        repr_pdf = repr_pdf.replace(group, '')
    
    return repr_pdf
