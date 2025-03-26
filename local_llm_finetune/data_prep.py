import openai
import os
import pandas as pd
import re


def replace_newpage_with_occurrence(text):
    # Define a function that replaces [newpage] with [newpage n]
    def replace_match(match, count=[0]):
        count[0] += 1  # Increment the count
        return f"[newpage {count[0]}]"

    # Use re.sub to replace all occurrences of [newpage]
    result = re.sub(r"\[newpage\]", replace_match, text)
    return result


def split_text_file(file_path, chunk_size, chunk_overlap):
    "split a text file into chunks"
    with open(file_path, "r") as f:
        text = f.read()

    # replace [newpage] with [newpage page_num]
    text = replace_newpage_with_occurrence(text)

    words = re.split(" ", text)
    chunks = []
    page_nums = []
    last_page_num = "NA"

    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

        pages = re.findall(r"\[newpage \d+\]", chunk)
        if len(pages) > 0:
            pages = [int(_.replace("[newpage", "").replace("]", "")) for _ in pages]
            last_page_num = pages[-1]
            if len(pages) > 1:
                page_num_text = f"{pages[0]}-{pages[1]}"
            else:
                page_num_text = str(pages[0])
        else:
            page_num_text = str(last_page_num)
        page_nums.append(page_num_text)

    return chunks, page_nums


def process_data(
    metadata,
    files_path,
    chunk_size,
    chunk_overlap,
    prompt_format="This is an excerpt from the document with the the following metadata: {}. It is currently about 500 words long. Summarize the information to about 100 words, keeping special note of key figures, statistics, and policy recommendations made. Here is the excerpt:\n\n'{}'",
    llm_url=None,
    llm_model=None,  # name of model for cloud providers
    api_key=None,
    completion_ratio=0.75,
):
    "process a corpus into a dataset"

    meta = []
    chunks = []
    file_paths = [_ for _ in os.listdir(files_path) if ".txt" in _]
    counter = 0

    for file_name in file_paths:
        print(f"chunking file {counter+1}/{len(file_paths)}")
        counter += 1

        doc_metadata = metadata.loc[
            lambda x: x["filepath"] == file_name, :
        ].reset_index(drop=True)
        meta_string = " | ".join(
            f"{col}: {val}"
            for col, val in zip(doc_metadata.columns, doc_metadata.iloc[0, :])
            if col not in ["filepath", "text_id", "vector_weight"]
        )

        tmp_chunks, page_nums = split_text_file(
            f"{files_path}{file_name}", chunk_size, chunk_overlap
        )

        chunks += tmp_chunks
        meta += [f"{meta_string} | page(s): {_}" for _ in page_nums]

    dataset = pd.DataFrame(
        {
            "query": (
                [prompt_format.format(meta[j], chunks[j]) for j in range(len(meta))]
                if llm_url is not None
                else [
                    prompt_format.format(
                        meta[j],
                        " ".join(
                            chunks[j].split(" ")[
                                : int(len(chunks[j].split(" ")) * completion_ratio)
                            ]
                        ),
                    )
                    for j in range(len(meta))
                ]
            ),
            "response": (
                ""
                if llm_url is not None
                else [
                    " ".join(
                        chunks[j].split(" ")[
                            int(len(chunks[j].split(" ")) * completion_ratio) :
                        ]
                    )
                    for j in range(len(meta))
                ]
            ),
        }
    )

    # LLM generated
    if llm_url is not None:
        client = openai.OpenAI(
            base_url=llm_url,
            api_key=api_key,
        )

        for i in range(5):  # range(len(dataset)):
            print(f"generating llm response: {i}/{len(dataset)}")

            response = (
                client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": dataset.loc[i, "query"]}],
                    temperature=0.0,
                    max_tokens=int(chunk_size * 1.25),
                )
                .choices[0]
                .message.content
            )
            dataset.loc[i, "response"] = response

    return dataset
