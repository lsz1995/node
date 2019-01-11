# -*- coding: utf-8 -*-
# python-docx-template �ĵ� https://docxtpl.readthedocs.io/en/latest/
# https://github.com/elapouya/python-docx-template/tree/master/tests
# https://github.com/elapouya/python-docx-template/blob/master/tests/inline_image.py

from docxtpl import DocxTemplate, InlineImage
# for height and width you have to use millimeters (Mm), inches or points(Pt) class :
from docx.shared import Mm, Inches, Pt
import jinja2


# #####################
# tpl = DocxTemplate("templates\\my_word_template.docx")  # ѡ��ʹ�õ�.docxģ��
# context = {'company_name': "World company"}  # company_name �Ǵ�����my_word_template.docx�ĵ�����ı�������������{{company_name}}��ֱ�ӷ���my_word_template.docx�ļ�����ȷλ�þ���
# tpl.render(context)  # ��������jinjia2��ģ�����Խ��б������滻��Ȼ��������������ĵ�generated_doc.docx���濴��{{company_name}}�����World company
# tpl.save("output\\generated_doc.docx")  # ����
# #####################

# #####################
# from docx import Document
# tpl = DocxTemplate("templates\\my_word_template.docx")  # ѡ��ʹ�õ�.docxģ��
# sub = tpl.new_subdoc()
# sub.subdocx = Document('templates\\subdocx.docx')  # Ҫ������ĵ�
# context = {'sub': sub}  # ������ĵ����ݷ���tpl��{{sub}}λ��
# tpl.render(context)
# tpl.save("output\\generated_doc.docx")  # ����
# #####################


tpl=DocxTemplate('templates/inline_image_tpl.docx')  # ѡ��ʹ�õ�.docxģ��

context = {
    'myimage' : InlineImage(tpl,'templates/python_logo.png',width=Mm(20)),
    'myimageratio': InlineImage(tpl, 'templates/python_jpeg.jpg', width=Mm(30), height=Mm(60)),

    'frameworks' : [{'image' : InlineImage(tpl,'templates/django.png',height=Mm(10)),
                      'desc' : 'The web framework for perfectionists with deadlines'},

                    {'image' : InlineImage(tpl,'templates/zope.png',height=Mm(10)),
                     'desc' : 'Zope is a leading Open Source Application Server and Content Management Framework'},

                    {'image': InlineImage(tpl, 'templates/pyramid.png', height=Mm(10)),
                     'desc': 'Pyramid is a lightweight Python web framework aimed at taking small web apps into big web apps.'},

                    {'image' : InlineImage(tpl,'templates/bottle.png',height=Mm(10)),
                     'desc' : 'Bottle is a fast, simple and lightweight WSGI micro web-framework for Python'},

                    {'image': InlineImage(tpl, 'templates/tornado.png', height=Mm(10)),
                     'desc': 'Tornado is a Python web framework and asynchronous networking library.'},
                    ]
}
# testing that it works also when autoescape has been forced to True
jinja_env = jinja2.Environment(autoescape=True)  # ת��=True
tpl.render(context, jinja_env)
tpl.save('output/inline_image.docx')