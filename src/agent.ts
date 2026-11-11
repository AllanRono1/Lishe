import { ChatOpenAI } from "@langchain/openai"
import { ToolMessage, type BaseMessageLike } from "@langchain/core/messages"
import { tavily } from "@tavily/core"
import * as dotenv from "dotenv"

//used to read content of .env file and loads its key-value pair into the
//  process.env so that the rest of the code can read them

dotenv.config()

if (!process.env.OPENROUTER_API_KEY) {
    throw new Error("OPENROUTER_API_KEY is not set")
}
if(!process.env.TAVILY_API_KEY) {
    throw new Error("TAVILY_API_KEY is not set")
}
//save the tavily api key as a key-value pair inside an object

const tavilyClient = tavily({apiKey: process.env.TAVILY_API_KEY})

// Connect to Google's Gemma 4 31B through OpenRouter. The ":free" at the end
// of the model name is what selects the free version; without it OpenRouter
// charges for each request. Created once here so that both graphs share the
// same model. Any replacement must support tool calling, because the
// researcher relies on it to request searches.
export const model = new ChatOpenAI({
    configuration: {baseURL: "https://openrouter.ai/api/v1"},
    modelName: "google/gemma-4-26b-a4b-it:free",
    apiKey: process.env.OPENROUTER_API_KEY
})


// Shared building blocks for the report graph in index.ts: the model, the
// search tools and the ReAct researcher loop.

/**
 * Removes duplicate and near-duplicate snippets from a list of search result
 * content strings. This matters here because several of the researcher's
 * queries overlap in topic (e.g. "renal restrictions" and "drug-nutrient
 * interactions" can both surface the same source page), so Tavily can return
 * the same snippet more than once across different queries.
 *
 * @param snippets - Raw content strings pulled from Tavily search results,
 *   one per hit, possibly containing duplicates across queries.
 * @returns The snippets with duplicates removed, preserving the order of
 *   first appearance.
 */
function dedupeSnippets(snippets: string[]): string[] {
    // Tracks a normalized version of every snippet we've already kept, so we
    // can check for repeats in O(1) instead of comparing every pair.
    const seen = new Set<string>()
    //stores the final, unique snippets to return to the user
    const deduped: string[] = []

    for (const snippet of snippets) {
        // Normalize before comparing: lowercase + collapsed whitespace means
        // two snippets that differ only by capitalization or line breaks are
        // still treated as the same underlying content.
        // const key = snippet.trim().toLowerCase().replace(/\s+/g, " ");
        const key = snippet.trim().toLowerCase().replace(/\s+/g, " ")
        if (key && !seen.has(key)) {
            seen.add(key);
            deduped.push(snippet.trim());
        }
    }

    return deduped;
}

// Tavily rejects search queries longer than 400 characters.
const MAX_SEARCH_QUERY_LENGTH = 400

// How many results each search asks Tavily for.
const RESULTS_PER_SEARCH = 3

// The most times the researcher may go round its loop (think, search, read
// the results) for one question or report. The search limit normally ends
// the loop first; this is a backstop in case the model keeps asking for
// searches that are never run (for example with an empty query).
const MAX_RESEARCH_ROUNDS = 6

// The Kenyan search only looks at these sites. A normal web search for a
// food question mostly returns American and British pages, which describe
// foods and portions a Kenyan patient may not recognise. These are the
// Kenyan government and research bodies that publish nutrition guidance
// (for example the Ministry of Health's National Guidelines for Healthy
// Diets and the Kenya Food Composition Tables), plus the FAO, which
// co-publishes several of them.
const KENYAN_DOMAINS = [
    "health.go.ke", //Ministry of Health
    "familyhealth.go.ke", //Ministry of Health, Division of Family Health
    "kilimo.go.ke", //Ministry of Agriculture
    "kemri.go.ke", //Kenya Medical Research Institute
    "fao.org", //Food and Agriculture Organization
]

// Put on the first line of every snippet that came from the Kenyan search,
// so the author can tell Kenyan sources apart from the rest.
export const KENYAN_LABEL = "[Kenyan source]"

// Tavily scores every result from 0 (unrelated) to 1 (a close match). The
// Kenyan search is limited to a handful of sites, so when none of them
// covers the question it still returns its "best" pages, which can be about
// something else entirely (a farming factsheet for a warfarin question).
// Results below this score are dropped, so that an unrelated page is never
// passed to the author as Kenyan evidence.
const MIN_KENYAN_SCORE = 0.4

// The tools the researcher can call, written in the format the model
// expects. The model never runs a search itself: it replies with a "tool
// call" naming one of these tools and the query it wants, and `research`
// runs the search and sends the results back to it.
export const WEB_SEARCH_TOOL = "web_search"
export const KENYAN_SEARCH_TOOL = "search_kenyan_sources"

const SEARCH_TOOLS = [
    {
        type: "function" as const,
        function: {
            name: WEB_SEARCH_TOOL,
            description: "Search the whole web. Returns the title, link and text of the top results.",
            parameters: {
                type: "object",
                properties: {
                    query: { 
                        type: "string", 
                        description: "A specific search query that makes sense on its own."
                     },
                },
                required: ["query"],
            },
        },
    },
    {
        type: "function" as const,
        function: {
            name: KENYAN_SEARCH_TOOL,
            description: "Search only Kenyan government and research sites (Ministry of Health, Ministry of Agriculture, KEMRI) and the FAO. Returns the title, link and text of the top results.",
            // Must be "parameters" (plural): with any other name the model is
            // never told this tool takes a query, so it calls it with none.
            parameters: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "A specific search query that makes sense on its own"
                    }
                },
                required: ["query"]
            }
        }
    },
]

// The same model, but told about the search tools, so it can ask for them.
const researcherModel = model.bindTools(SEARCH_TOOLS)

/**
 * Runs one Tavily search and turns the results into snippets for the
 * author. Each snippet keeps its title and link in front of its text, so
 * the author can tell the reader where a fact came from. Snippets from the
 * Kenyan search also get KENYAN_LABEL as their first line, and are kept only
 * if they are relevant enough (see MIN_KENYAN_SCORE).
 *
 * @param query - What to search for.
 * @param kenyan - Whether to search only the Kenyan sites.
 * @returns The snippets, best match first. Empty if nothing was found.
 */
async function search(query: string, kenyan: boolean): Promise<string[]> {
    const response = await tavilyClient.search(query.slice(0, MAX_SEARCH_QUERY_LENGTH), {
        maxResults: RESULTS_PER_SEARCH,
        ...(kenyan ? { includeDomains: KENYAN_DOMAINS } : {}),
    })
    const label = kenyan ? `${KENYAN_LABEL}\n` : ""
    return response.results
        ?.filter((res) => !kenyan || res.score >= MIN_KENYAN_SCORE)
        .map((res) => `${label}${res.title}\nLink: ${res.url}\n${res.content.trim()}`) || []
}

/**
 * The researcher, as a ReAct agent ("reason + act"). Both graphs' researcher
 * nodes call this; they only differ in the instructions they give it.
 *
 * It works in rounds:
 *   1. Reason: the model reads the task (and every result so far) and
 *      decides what is still missing. It answers with one or more tool
 *      calls, each one a search it wants run.
 *   2. Act: we run those searches (at the same time) and send the results
 *      back to the model as "tool" messages.
 *   3. Repeat, until the model stops asking for searches because it has
 *      enough, or the search limit is reached.
 *
 * The model's own final words are not used: the author writes from the
 * snippets the searches returned, not from the researcher's summary of them.
 *
 * The loop runs inside this one function, rather than as a loop of nodes in
 * the graph, so its working notes (the tool calls and results) are thrown
 * away when it finishes. In the chat graph, anything in the graph state is
 * saved between messages, and these notes are not worth keeping.
 *
 * One search is always kept back for the Kenyan sites. If the model never
 * searched them itself, its first web query is run again there, so the
 * author always has local guidance to draw on (or knows there was none).
 *
 * @param instructions - The system prompt: what to research and how.
 * @param conversation - The messages that state the task (for the chat, the
 *   recent conversation followed by the new question).
 * @param maxSearches - The most searches to run, including the Kenyan one.
 *   Every search is one Tavily request, so this is also the most it costs.
 * @param fallbackQuery - What to search for if the model fails before it
 *   searches anything, so the author still has something to work from.
 * @param maxRounds - The most times to ask the model what to search. Every
 *   round is one model call, and each can take 20 to 30 seconds on the free
 *   model, so the chat uses 1: the model plans all its searches at once, and
 *   we skip the extra call where it reads the results and usually just says
 *   it has enough. The report keeps several rounds, since it is not waited
 *   on live.
 * @returns The deduplicated snippets from every search that was run. Empty
 *   if the model decided no search was needed.
 */
export async function research(
    instructions: string,
    conversation: BaseMessageLike[],//(union type - allows loose and flexible shorthand inputs) rep an array of objects that can be converted to chat messages
    maxSearches: number,
    fallbackQuery: string,
    maxRounds: number = MAX_RESEARCH_ROUNDS
): Promise<string[]> {
    // Everything the model has seen so far. It grows each round with the
    // model's tool calls and the results we send back.
    const messages: BaseMessageLike[] = [{ role: "system", content: instructions }, ...conversation]

    const snippets: string[] = []
    const searchesRun: { query: string, kenyan: boolean }[] = []
    let searchesLeft = maxSearches - 1 //one is kept back for the Kenyan search
    let modelFailed = false

    for (let round = 0; round < maxRounds && searchesLeft > 0; round++) {
        // 1. Reason: ask the model what to search for next.
        let response
        try {
            response = await researcherModel.invoke(messages)
        } catch (error) {
            console.warn("The researcher model failed:", error)
            modelFailed = true
            break
        }

        // No tool calls means the model has decided it has enough.
        const toolCalls = response.tool_calls ?? []
        if (toolCalls.length === 0) {
            break
        }
        messages.push(response)

        // 2. Act: run every search it asked for. Each call is checked first,
        // and given its place in the search limit, before any of them start.
        const results = await Promise.all(toolCalls.map(async (call) => {
            const query = typeof call.args.query === "string" ? call.args.query.trim() : ""
            const kenyan = call.name === KENYAN_SEARCH_TOOL

            if (call.name !== WEB_SEARCH_TOOL && !kenyan) {
                return `Not run: there is no tool called "${call.name}".`
            }
            if (!query) {
                return "Not run: the query was empty."
            }
            if (searchesLeft === 0) {
                return "Not run: the search limit has been reached."
            }
            searchesLeft--
            searchesRun.push({ query, kenyan })
            console.log(`Search (${call.name}):`, query)

            try {
                const found = await search(query, kenyan)
                snippets.push(...found)
                return found.join("\n\n") || "No relevant results were found."
            } catch (error) {
                console.warn(`Search failed for query "${query}":`, error)
                return "The search failed."
            }
        }))

        // 3. Send the results back, so the model can read them next round.
        // Each result has to name the tool call it answers.
        toolCalls.forEach((call, i) => {
            messages.push(new ToolMessage({ tool_call_id: call.id ?? "", content: results[i] ?? "" }))
        })
    }

    // If the model broke before it could search anything, search for the
    // task exactly as it was given, so the author still has an answer to
    // work from.
    if (modelFailed && searchesRun.length === 0) {
        searchesRun.push({ query: fallbackQuery, kenyan: false })
        console.log(`Search (${WEB_SEARCH_TOOL}, fallback):`, fallbackQuery)
        try {
            snippets.push(...await search(fallbackQuery, false))
        } catch (error) {
            console.warn(`Search failed for query "${fallbackQuery}":`, error)
        }
    }

    // The search that was kept back: if the model searched the web but never
    // the Kenyan sites, run its first web query again on them. Room is left
    // for the word "Kenya" so that adding it never pushes the query over
    // Tavily's length limit.
    const firstWebSearch = searchesRun.find((s) => !s.kenyan)
    if (firstWebSearch && !searchesRun.some((s) => s.kenyan)) {
        const kenyanQuery = `${firstWebSearch.query.slice(0, MAX_SEARCH_QUERY_LENGTH - " Kenya".length)} Kenya`
        console.log(`Search (${KENYAN_SEARCH_TOOL}):`, kenyanQuery)
        try {
            snippets.push(...await search(kenyanQuery, true))
        } catch (error) {
            console.warn(`Search failed for query "${kenyanQuery}":`, error)
        }
    }

    // Different searches often surface the same source page; collapse those
    // before handing the context to the author.
    return dedupeSnippets(snippets)
}
