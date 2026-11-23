import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    n = len(corpus)
    distribution = {}

    # Base: with probability (1 - d), choose any page uniformly
    for p in corpus:
        distribution[p] = (1 - damping_factor) / n

    links = corpus[page]

    if links:
        # With probability d, spread d evenly over outgoing links
        for link in links:
            distribution[link] += damping_factor / len(links)
    else:
        # No outgoing links: pretend page links to everyone uniformly
        for p in corpus:
            distribution[p] += damping_factor / n

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize counts
    counts = {page: 0 for page in corpus}

    # First sample: uniformly random
    current = random.choice(list(corpus.keys()))
    counts[current] += 1

    # Remaining n-1 samples: follow the transition model
    for _ in range(n - 1):
        distribution = transition_model(corpus, current, damping_factor)

        # Weighted random choice based on probabilities
        pages = list(distribution.keys())
        weights = [distribution[p] for p in pages]
        current = random.choices(pages, weights=weights, k=1)[0]
        counts[current] += 1

    # Convert counts to proportions
    return {page: counts[page] / n for page in corpus}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)

    # Start with uniform ranks: every page gets 1 / N
    ranks = {page: 1 / n for page in corpus}

    while True:
        new_ranks = {}

        for page in corpus:
            # Base term: (1 - d) / N
            rank = (1 - damping_factor) / n

            # Sum contributions from all pages that link to `page`
            for potential_linker in corpus:
                links = corpus[potential_linker]

                if page in links:
                    # `potential_linker` links to `page`:
                    # contribute d * PR(linker) / NumLinks(linker)
                    rank += damping_factor * ranks[potential_linker] / len(links)
                elif not links:
                    # No outgoing links: treat as linking to everyone,
                    # so it contributes d * PR(linker) / N to every page
                    rank += damping_factor * ranks[potential_linker] / n

            new_ranks[page] = rank

        # Check convergence: no rank changed by more than 0.001
        max_change = max(abs(new_ranks[p] - ranks[p]) for p in corpus)
        ranks = new_ranks

        if max_change < 0.001:
            break

    return ranks


if __name__ == "__main__":
    main()
