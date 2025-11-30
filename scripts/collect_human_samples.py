"""
Continuous Human Text Collector for VERITAS - EXPANDED EDITION

This script scrapes genuine human-written text from various internet sources
and saves them to the training dataset. Runs continuously until stopped.

Sources (MASSIVELY EXPANDED):
- Reddit (100+ diverse subreddits covering Q&A, tech, science, writing, arts,
  professional, education, philosophy, history, health, gaming, news, hobbies,
  and personal stories)
- Hacker News (posts and comments from tech discussions)
- Stack Overflow (65+ technology tags covering languages, web tech, databases,
  mobile, data science, DevOps, and more)
- ArXiv (45+ academic categories in CS, statistics, math, physics, quantitative
  biology, finance, and economics)
- Wikipedia (random articles across all topics)
- Project Gutenberg (classic literature excerpts from public domain books)
- Famous Quotes (quotations from historical figures)

Usage:
    python scripts/collect_human_samples.py [--max-samples 200000] [--output data/human_samples.json]

Press Ctrl+C to stop gracefully.
"""

import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import random


class HumanTextCollector:
    """Collects genuine human text from various internet sources."""

    def __init__(self, output_file: str = "data/human_samples.json", max_samples: int = 1000):
        self.output_file = Path(output_file)
        self.max_samples = max_samples
        self.samples = []
        self.collected_ids = set()  # Avoid duplicates

        # Load existing samples if file exists
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                self.samples = existing if isinstance(existing, list) else []
                self.collected_ids = {s.get('id', '') for s in self.samples}
                print(f"[LOAD] Loaded {len(self.samples)} existing samples")

        # Rate limiting
        self.last_request_time = {}

    def rate_limit(self, source: str, min_delay: float = 1.0):
        """Ensure minimum delay between requests to same source."""
        if source in self.last_request_time:
            elapsed = time.time() - self.last_request_time[source]
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
        self.last_request_time[source] = time.time()

    def save_samples(self):
        """Save samples to disk."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.samples, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] Saved {len(self.samples)} samples to {self.output_file}")

    def add_sample(self, text: str, source: str, metadata: Dict):
        """Add a sample if it meets quality criteria."""
        # Quality checks
        if len(text) < 100:  # Too short
            return False

        if len(text) > 10000:  # Too long, truncate
            text = text[:10000]

        # Check for duplicates
        sample_id = metadata.get('id', f"{source}_{hash(text)}")
        if sample_id in self.collected_ids:
            return False

        # Add sample
        sample = {
            'id': sample_id,
            'text': text,
            'label': 'human',
            'source': source,
            'collected_at': datetime.now().isoformat(),
            'word_count': len(text.split()),
            'metadata': metadata
        }

        self.samples.append(sample)
        self.collected_ids.add(sample_id)
        return True

    def collect_reddit(self, subreddit: str = 'AskReddit', limit: int = 25) -> int:
        """Collect posts and comments from Reddit."""
        self.rate_limit('reddit', 2.0)

        try:
            # Use Reddit's JSON API (no auth required for public posts)
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
            headers = {'User-Agent': 'VERITAS-DataCollector/1.0'}

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            collected = 0

            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})

                # Collect post text (self posts)
                if post_data.get('selftext') and not post_data.get('is_self') is False:
                    text = post_data['selftext']
                    metadata = {
                        'id': post_data['id'],
                        'subreddit': subreddit,
                        'title': post_data.get('title', ''),
                        'score': post_data.get('score', 0),
                        'author': post_data.get('author', '[deleted]'),
                        'type': 'reddit_post'
                    }

                    if self.add_sample(text, f'reddit_{subreddit}', metadata):
                        collected += 1
                        print(f"  [+] Reddit post from r/{subreddit} ({len(text)} chars)")

            return collected

        except Exception as e:
            print(f"  [!] Reddit error: {e}")
            return 0

    def collect_hackernews(self, limit: int = 30) -> int:
        """Collect comments from Hacker News."""
        self.rate_limit('hackernews', 1.0)

        try:
            # Get top stories
            url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            story_ids = response.json()[:limit]
            collected = 0

            for story_id in story_ids[:10]:  # Limit to first 10 to avoid too many requests
                self.rate_limit('hackernews', 1.0)

                # Get story details
                story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                story_response = requests.get(story_url, timeout=10)

                if story_response.status_code != 200:
                    continue

                story = story_response.json()

                # Collect text from story (if it's a text post)
                if story.get('text'):
                    metadata = {
                        'id': f"hn_{story_id}",
                        'title': story.get('title', ''),
                        'score': story.get('score', 0),
                        'author': story.get('by', 'unknown'),
                        'type': 'hackernews_post'
                    }

                    if self.add_sample(story['text'], 'hackernews', metadata):
                        collected += 1
                        print(f"  [+] HN post #{story_id} ({len(story['text'])} chars)")

                # Collect comments
                if story.get('kids'):
                    for comment_id in story['kids'][:5]:  # First 5 comments
                        self.rate_limit('hackernews', 0.5)

                        comment_url = f"https://hacker-news.firebaseio.com/v0/item/{comment_id}.json"
                        comment_response = requests.get(comment_url, timeout=10)

                        if comment_response.status_code != 200:
                            continue

                        comment = comment_response.json()

                        if comment.get('text'):
                            metadata = {
                                'id': f"hn_comment_{comment_id}",
                                'story_id': story_id,
                                'author': comment.get('by', 'unknown'),
                                'type': 'hackernews_comment'
                            }

                            if self.add_sample(comment['text'], 'hackernews', metadata):
                                collected += 1
                                print(f"  [+] HN comment #{comment_id} ({len(comment['text'])} chars)")

                if collected >= limit:
                    break

            return collected

        except Exception as e:
            print(f"  [!] HackerNews error: {e}")
            return 0

    def collect_stackoverflow(self, tag: str = 'python', limit: int = 20) -> int:
        """Collect answers from Stack Overflow."""
        self.rate_limit('stackoverflow', 2.0)

        try:
            # Stack Exchange API (no key required for basic usage, but rate limited)
            url = f"https://api.stackexchange.com/2.3/questions"
            params = {
                'order': 'desc',
                'sort': 'activity',
                'tagged': tag,
                'site': 'stackoverflow',
                'filter': 'withbody',  # Include question body
                'pagesize': limit
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            collected = 0

            for question in data.get('items', []):
                # Collect question body
                if question.get('body'):
                    text = self._strip_html(question['body'])
                    metadata = {
                        'id': f"so_q_{question['question_id']}",
                        'title': question.get('title', ''),
                        'score': question.get('score', 0),
                        'tags': question.get('tags', []),
                        'author': question.get('owner', {}).get('display_name', 'unknown'),
                        'type': 'stackoverflow_question'
                    }

                    if self.add_sample(text, f'stackoverflow_{tag}', metadata):
                        collected += 1
                        print(f"  [+] SO question #{question['question_id']} ({len(text)} chars)")

                if collected >= limit:
                    break

            # Check rate limit
            quota_remaining = data.get('quota_remaining', 0)
            if quota_remaining < 100:
                print(f"  [!] SO API quota low: {quota_remaining} remaining")

            return collected

        except Exception as e:
            print(f"  [!] StackOverflow error: {e}")
            return 0

    def collect_arxiv(self, category: str = 'cs.AI', limit: int = 10) -> int:
        """Collect abstracts from ArXiv papers (human-written research)."""
        self.rate_limit('arxiv', 3.0)

        try:
            # ArXiv API
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'cat:{category}',
                'start': 0,
                'max_results': limit,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)

            collected = 0
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            for entry in root.findall('atom:entry', ns):
                abstract = entry.find('atom:summary', ns)
                title = entry.find('atom:title', ns)
                authors = entry.findall('atom:author', ns)
                arxiv_id = entry.find('atom:id', ns)

                if abstract is not None and abstract.text:
                    text = abstract.text.strip()
                    author_names = [a.find('atom:name', ns).text for a in authors if a.find('atom:name', ns) is not None]

                    metadata = {
                        'id': f"arxiv_{arxiv_id.text.split('/')[-1] if arxiv_id is not None else hash(text)}",
                        'title': title.text.strip() if title is not None else '',
                        'authors': author_names,
                        'category': category,
                        'type': 'arxiv_abstract'
                    }

                    if self.add_sample(text, f'arxiv_{category}', metadata):
                        collected += 1
                        print(f"  [+] ArXiv abstract from {category} ({len(text)} chars)")

            return collected

        except Exception as e:
            print(f"  [!] ArXiv error: {e}")
            return 0

    def collect_wikipedia(self, limit: int = 10) -> int:
        """Collect excerpts from Wikipedia articles."""
        self.rate_limit('wikipedia', 2.0)

        try:
            # Get random articles
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'random',
                'rnlimit': limit,
                'rnnamespace': 0  # Main namespace only
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            collected = 0

            for page in data.get('query', {}).get('random', []):
                page_id = page['id']
                page_title = page['title']

                # Get page content
                content_params = {
                    'action': 'query',
                    'format': 'json',
                    'pageids': page_id,
                    'prop': 'extracts',
                    'exintro': True,  # Get intro section only
                    'explaintext': True  # Plain text, no HTML
                }

                self.rate_limit('wikipedia', 1.0)
                content_response = requests.get(url, params=content_params, timeout=10)

                if content_response.status_code != 200:
                    continue

                content_data = content_response.json()
                pages = content_data.get('query', {}).get('pages', {})

                if str(page_id) in pages:
                    extract = pages[str(page_id)].get('extract', '')

                    if extract:
                        metadata = {
                            'id': f"wiki_{page_id}",
                            'title': page_title,
                            'type': 'wikipedia_article'
                        }

                        if self.add_sample(extract, 'wikipedia', metadata):
                            collected += 1
                            print(f"  [+] Wikipedia: {page_title} ({len(extract)} chars)")

            return collected

        except Exception as e:
            print(f"  [!] Wikipedia error: {e}")
            return 0

    def collect_gutenberg(self, limit: int = 5) -> int:
        """Collect excerpts from Project Gutenberg books (public domain literature)."""
        self.rate_limit('gutenberg', 3.0)

        try:
            # Popular public domain books
            book_ids = [
                1342,  # Pride and Prejudice
                84,    # Frankenstein
                1661,  # Sherlock Holmes
                11,    # Alice in Wonderland
                2701,  # Moby Dick
                1952,  # The Yellow Wallpaper
                16,    # Peter Pan
                76,    # Adventures of Huckleberry Finn
                98,    # A Tale of Two Cities
                244,   # A Study in Scarlet
            ]

            collected = 0

            for book_id in random.sample(book_ids, min(limit, len(book_ids))):
                try:
                    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
                    response = requests.get(url, timeout=15)

                    if response.status_code != 200:
                        continue

                    text = response.text

                    # Extract a random passage (avoid header/footer)
                    lines = text.split('\n')
                    start = random.randint(100, max(100, len(lines) - 500))
                    passage = '\n'.join(lines[start:start + 50])  # ~50 lines

                    if len(passage) > 200:
                        metadata = {
                            'id': f"gutenberg_{book_id}_{start}",
                            'book_id': book_id,
                            'type': 'literature_excerpt'
                        }

                        if self.add_sample(passage, 'gutenberg', metadata):
                            collected += 1
                            print(f"  [+] Gutenberg book #{book_id} excerpt ({len(passage)} chars)")

                    self.rate_limit('gutenberg', 2.0)

                except Exception as e:
                    print(f"  [!] Error with book {book_id}: {e}")
                    continue

            return collected

        except Exception as e:
            print(f"  [!] Gutenberg error: {e}")
            return 0

    def collect_quotes(self, limit: int = 20) -> int:
        """Collect famous quotes and their context."""
        self.rate_limit('quotes', 2.0)

        try:
            # ZenQuotes API (free, no auth required)
            url = "https://zenquotes.io/api/quotes"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return 0

            quotes = response.json()
            collected = 0

            for quote_data in quotes[:limit]:
                quote_text = quote_data.get('q', '')
                author = quote_data.get('a', '')

                if quote_text and len(quote_text) > 50:
                    full_text = f"{quote_text} - {author}"

                    metadata = {
                        'id': f"quote_{hash(quote_text)}",
                        'author': author,
                        'type': 'quote'
                    }

                    if self.add_sample(full_text, 'quotes', metadata):
                        collected += 1
                        print(f"  [+] Quote by {author} ({len(full_text)} chars)")

            return collected

        except Exception as e:
            print(f"  [!] Quotes error: {e}")
            return 0

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags from text."""
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Decode HTML entities
        import html as html_module
        text = html_module.unescape(text)
        return text.strip()

    def run(self):
        """Main collection loop."""
        print("="*60)
        print("VERITAS Human Text Collector")
        print("="*60)
        print(f"Target: {self.max_samples} samples")
        print(f"Current: {len(self.samples)} samples")
        print(f"Output: {self.output_file}")
        print("\nCollecting from:")
        print("  - Reddit (100+ diverse subreddits)")
        print("  - Hacker News (posts and comments)")
        print("  - Stack Overflow (65+ technology tags)")
        print("  - ArXiv (45+ academic categories)")
        print("  - Wikipedia (random articles)")
        print("  - Project Gutenberg (classic literature)")
        print("  - Quotes (famous quotations)")
        print("\nPress Ctrl+C to stop gracefully")
        print("="*60)

        # Diverse subreddits for varied human text - MASSIVELY EXPANDED
        subreddits = [
            # Q&A and Discussion
            'AskReddit', 'explainlikeimfive', 'AskScience', 'askphilosophy',
            'AskHistorians', 'NoStupidQuestions', 'TooAfraidToAsk', 'changemyview',

            # Technology & Programming
            'technology', 'programming', 'learnprogramming', 'python', 'javascript',
            'webdev', 'linux', 'datascience', 'machinelearning', 'artificial',
            'cybersecurity', 'devops', 'git', 'vim', 'emacs',

            # Science & Academia
            'science', 'physics', 'chemistry', 'biology', 'space', 'astronomy',
            'math', 'statistics', 'neuroscience', 'psychology', 'AcademicPsychology',

            # Writing & Literature
            'writing', 'WritingPrompts', 'books', 'literature', 'poetry',
            'scifiwriting', 'fantasywriters', 'booksuggestions', 'suggestmeabook',

            # Creative & Arts
            'Art', 'Design', 'photography', 'Music', 'WeAreTheMusicMakers',
            'gamedev', 'worldbuilding', 'conlang',

            # Professional & Career
            'cscareerquestions', 'ITCareerQuestions', 'careerguidance',
            'resumes', 'startups', 'Entrepreneur', 'smallbusiness',

            # Education & Learning
            'todayilearned', 'YouShouldKnow', 'LifeProTips', 'GetStudying',
            'StudyTips', 'HomeworkHelp', 'learnmath', 'languagelearning',

            # Philosophy & Ideas
            'philosophy', 'stoicism', 'Existentialism', 'ethics', 'DebateReligion',
            'TrueAtheism', 'religion', 'spirituality',

            # History & Culture
            'history', 'HistoryMemes', 'anthropology', 'AskAnthropology',
            'linguistics', 'etymology', 'badhistory', 'HistoricalWhatIf',

            # Health & Fitness
            'fitness', 'nutrition', 'AdvancedFitness', 'bodyweightfitness',
            'running', 'cycling', 'yoga', 'mentalhealth',

            # Gaming
            'gaming', 'Games', 'truegaming', 'patientgamers', 'gamedesign',

            # News & Politics
            'news', 'worldnews', 'NeutralPolitics', 'PoliticalDiscussion',
            'geopolitics', 'economics', 'Economics',

            # Hobbies & Interests
            'cooking', 'food', 'recipes', 'AskCulinary', 'gardening',
            'DIY', 'woodworking', 'HomeImprovement', 'travel', 'solotravel',

            # Personal Stories & Experiences
            'confession', 'offmychest', 'CasualConversation', 'self',
            'TrueOffMyChest', 'unpopularopinion', 'Showerthoughts'
        ]

        # Stack Overflow tags - MASSIVELY EXPANDED
        so_tags = [
            # Popular Languages
            'python', 'javascript', 'java', 'c#', 'c++', 'php', 'ruby', 'go',
            'rust', 'swift', 'kotlin', 'typescript', 'r', 'scala', 'perl',

            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'express',
            'django', 'flask', 'spring', 'asp.net', 'laravel', 'jquery',

            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle',
            'sql-server', 'database', 'redis', 'elasticsearch',

            # Mobile & Desktop
            'android', 'ios', 'flutter', 'react-native', 'xamarin',

            # Data & ML
            'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn',
            'machine-learning', 'deep-learning', 'neural-network',

            # DevOps & Tools
            'docker', 'kubernetes', 'git', 'jenkins', 'azure', 'aws',
            'google-cloud-platform', 'linux', 'bash', 'powershell',

            # Other
            'algorithms', 'data-structures', 'regex', 'json', 'xml',
            'api', 'rest', 'graphql', 'websocket', 'oauth'
        ]

        # ArXiv categories - MASSIVELY EXPANDED
        arxiv_categories = [
            # Computer Science
            'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.CR', 'cs.DS', 'cs.DB',
            'cs.DC', 'cs.SE', 'cs.NE', 'cs.PL', 'cs.HC', 'cs.IR', 'cs.RO',
            'cs.SI', 'cs.SY', 'cs.GT', 'cs.CG', 'cs.IT',

            # Statistics & Math
            'stat.ML', 'stat.ME', 'stat.TH', 'stat.AP', 'stat.CO',
            'math.ST', 'math.CO', 'math.OC', 'math.NA', 'math.PR',

            # Physics
            'physics.comp-ph', 'physics.data-an', 'physics.soc-ph',

            # Quantitative Fields
            'q-bio.QM', 'q-bio.NC', 'q-fin.ST', 'q-fin.CP',

            # Economics
            'econ.EM', 'econ.TH'
        ]

        iteration = 0

        try:
            while len(self.samples) < self.max_samples:
                iteration += 1
                print(f"\n[Iteration {iteration}] Samples: {len(self.samples)}/{self.max_samples}")

                batch_collected = 0

                # Randomly select which sources to use this iteration for variety
                sources = ['reddit', 'hackernews', 'stackoverflow', 'arxiv', 'wikipedia', 'gutenberg', 'quotes']
                random.shuffle(sources)

                # Use 4-5 random sources per iteration
                for source in sources[:random.randint(4, 5)]:
                    if source == 'reddit':
                        subreddit = random.choice(subreddits)
                        print(f"\n[Reddit] Collecting from r/{subreddit}...")
                        batch_collected += self.collect_reddit(subreddit, limit=15)

                    elif source == 'hackernews':
                        print(f"\n[HackerNews] Collecting posts and comments...")
                        batch_collected += self.collect_hackernews(limit=10)

                    elif source == 'stackoverflow':
                        tag = random.choice(so_tags)
                        print(f"\n[StackOverflow] Collecting from tag '{tag}'...")
                        batch_collected += self.collect_stackoverflow(tag, limit=10)

                    elif source == 'arxiv':
                        category = random.choice(arxiv_categories)
                        print(f"\n[ArXiv] Collecting from {category}...")
                        batch_collected += self.collect_arxiv(category, limit=5)

                    elif source == 'wikipedia':
                        print(f"\n[Wikipedia] Collecting random articles...")
                        batch_collected += self.collect_wikipedia(limit=8)

                    elif source == 'gutenberg':
                        print(f"\n[Gutenberg] Collecting literature excerpts...")
                        batch_collected += self.collect_gutenberg(limit=3)

                    elif source == 'quotes':
                        print(f"\n[Quotes] Collecting famous quotes...")
                        batch_collected += self.collect_quotes(limit=15)

                # Save progress
                if batch_collected > 0:
                    self.save_samples()
                    print(f"\n[PROGRESS] Total collected: {len(self.samples)}/{self.max_samples} ({batch_collected} this iteration)")
                else:
                    print(f"\n[WARNING] No new samples collected this iteration")

                # Check if we've reached the target
                if len(self.samples) >= self.max_samples:
                    print(f"\n[COMPLETE] Reached target of {self.max_samples} samples!")
                    break

                # Sleep between iterations to be respectful to APIs
                sleep_time = 10
                print(f"\n[SLEEP] Waiting {sleep_time} seconds before next iteration...")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n\n[STOP] Interrupted by user")

        finally:
            # Final save
            self.save_samples()
            print(f"\n[FINAL] Collected {len(self.samples)} total samples")
            print(f"[FINAL] Saved to {self.output_file}")

            # Statistics
            sources = {}
            for sample in self.samples:
                source = sample['source']
                sources[source] = sources.get(source, 0) + 1

            print("\n[STATS] Samples by source:")
            for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                print(f"  {source}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Collect human text samples for VERITAS training')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum samples to collect')
    parser.add_argument('--output', type=str, default='data/human_samples.json', help='Output file')

    args = parser.parse_args()

    collector = HumanTextCollector(
        output_file=args.output,
        max_samples=args.max_samples
    )

    collector.run()


if __name__ == "__main__":
    main()
