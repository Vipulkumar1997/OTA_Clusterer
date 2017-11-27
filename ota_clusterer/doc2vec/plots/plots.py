import matplotlib.pyplot as plt

# TODO


def create_most_similar_doc_plot(most_similar_doc_matrix):
    fig, ax = plt.subplots()
    plt.scatter(most_similar_doc_matrix[:0], most_similar_doc_matrix[:1])
    plt.show(block=True)