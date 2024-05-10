import pandas as pd
import re


def split_text_file(file_path, chunk_size, chunk_overlap):
    "split a text file into chunks"
    with open(file_path, "r") as f:
        text = f.read()

    words = re.split(" ", text)
    chunks = []

    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    return chunks


def process_data(
    metadata,
    chunk_size,
    chunk_overlap,
    prompt_format="This is an excerpt from the document with the the following metadata: {}. It is currently about 500 words long. Summarize the information to about 100 words, keeping special note of key figures, statistics, and policy recommendations made. Here is the excerpt: '{}'",
    llm=None,
    model=None,
    completion_ratio=0.75,
):
    "process a corpus into a dataset"

    meta = []
    chunks = []
    for i in range(len(metadata)):
        print(f"chunking file {i+1}/{len(metadata)}")
        meta_string = ""
        for col in metadata.columns:
            if col != "filepath":
                meta_string += f"'{col}': {metadata.loc[i, col]}" + (
                    " | " if col != metadata.columns[-1] else ""
                )
        tmp_chunks = split_text_file(
            metadata.loc[i, "filepath"], chunk_size, chunk_overlap
        )
        chunks += tmp_chunks
        meta += [meta_string] * len(tmp_chunks)

    dataset = pd.DataFrame(
        {
            "query": [
                prompt_format.format(meta[j], chunks[j]) for j in range(len(meta))
            ]
            if llm != None
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
            ],
            "response": ""
            if llm != None
            else [
                " ".join(
                    chunks[j].split(" ")[
                        int(len(chunks[j].split(" ")) * completion_ratio) :
                    ]
                )
                for j in range(len(meta))
            ],
        }
    )

    # LLM generated
    if llm != None:
        for i in range(len(dataset)):
            print(f"generating responses {i}/{len(dataset)}")
            response = model.gen_response(
                prompt=dataset.loc[i, "query"],
                llm=llm,
                use_chat_engine=False,
            )
            dataset.loc[i, "response"] = response["response"]

    return dataset


def clean_up_response(response, model_family="llama-3"):
    "clean up responses from an LLM to get rid of extraneous information"

    if model_family == "llama-3":
        # remove "here is a summary text
        if "here is a summary" in response.split("\n\n")[0].lower():
            response = "\n\n".join(response.split("\n\n")[1:])

        # remove anything after another system: assistant: exchange
        response = response.split("system:")[0]
        response = response.split("assistant:")[0]

        # remove any paragraphs that end with an exclamation point, unnecessary
        response = "\n\n".join(
            [x for x in response.split("\n\n") if "!" not in x[-10:]]
        )

    return response
