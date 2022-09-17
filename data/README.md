# Handbook QA: question-answer pairs from PGT handbook.
The data is a list of dictionaries. Each dictionary stores the information of a section in the PGT handbook.
Since there are 47 sections in the handbook, there are 47 dictionaries in the list.

- For each `section` dictionary, there are three keys:
    - `'section_id'` stores the id of the section.
    - `'section_name'` stores the title of the section.
    - `'paragraphs'` is a list of dictionaries. Each dictionary stores the information of a paragraph in this section.

- For each `paragraph` dictionary, there are keys:
    - `'paragraph_id'` stores the id of the paragraph.
    - `'context'` stores the content of the paragraph.
    - `'qas'` is a list of dictionaries. Each dictionary stores the information of a original question with all its reformulation questions related to this paragraph.

- For each `qas` dictionary, there are five keys:
    - `'answer_start'` stores where the answer starts in the paragraph.
    - `'answer_text'` stores the text of the answer span.
    - `'question'` stores the text of the question.
    - `'question_id'` stores the id of the question.
    - `'type'` is a dictionary storing the original source of the question.
    
- There are three keys in the `type` dictionary:
    - `'meaning_type'`: is it meaningful or meaningless
    - `'original_type'`: is the original question generated from human or QG model
    - `'reformulation_type'`: is it a original question or a paraphrase question or a noisy question