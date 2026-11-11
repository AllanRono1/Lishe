import { StateGraph, Annotation, END } from "@langchain/langgraph"
import { writeFile } from "node:fs/promises"
import { pathToFileURL } from "node:url"
import { marked } from "marked"
import { model, research, WEB_SEARCH_TOOL, KENYAN_SEARCH_TOOL, KENYAN_LABEL } from "./agent.js"

// The REPORT graph (researcher -> author). The website (server.ts) calls
// `generateReport`; running this file directly (`npm run report`) writes one
// report to report.md / report.html from the terminal.

// ---------------------------------------------------------------------------
// Report types: each one is a layout the author must follow.
// ---------------------------------------------------------------------------

export type ReportType = {
    label: string
    description: string
    structure: string
}

export const REPORT_TYPES = {
    comprehensive: {
        label: "Comprehensive clinical report",
        description: "Every patient group, restrictions, interactions, a meal plan and sources.",
        structure: `This article should be a practical clinical reference, structured as:
1. Introduction - why nutrition matters for immunocompromised patients and how it relates to the topic (1-2 paragraphs)
2. One section per patient group (elderly, RVD (Revlimid/Velcade/dexamethasone), HIV, diabetes, cancer, infants and neonates, post-operative), each with key nutritional needs and foods to favour or avoid
3. Food safety for immunocompromised patients (neutropenic precautions, safe food handling)
4. Renal and hepatic dietary restrictions
5. Drug-nutrient interactions - a Markdown table of drug/class, food or nutrient, effect, and advice
6. Sample Kenyan meal plan - a Markdown table covering breakfast, lunch, supper and snacks
7. Conclusion - key safety takeaways as bullet points
8. Sources`,
    },
    mealPlan: {
        label: "Kenyan meal plan",
        description: "A 7-day plan built from affordable Kenyan foods, with safety notes.",
        structure: `This article should be a practical meal-planning guide, structured as:
1. Introduction - nutritional goals for the patient group in the topic (1 paragraph)
2. Key nutrients and target foods - bullet list of affordable Kenyan sources for each
3. Foods to avoid or limit, and why (food safety, interactions, renal/hepatic limits)
4. 7-day Kenyan meal plan - a Markdown table with columns Day, Breakfast, Mid-morning, Lunch, Afternoon snack, Supper
5. Preparation and food-safety tips for immunocompromised patients
6. Adjustments for swallowing difficulty, poor appetite, diabetes and kidney disease
7. Sources`,
    },
    drugInteractions: {
        label: "Drug-nutrient interactions",
        description: "Interaction tables and timing advice for the medicines in the topic.",
        structure: `This article should be a drug-nutrient interaction reference, structured as:
1. Introduction - why drug-nutrient interactions matter for immunocompromised patients (1 paragraph)
2. Interaction table - a Markdown table with columns Drug/class, Food or nutrient, Mechanism (e.g. CYP3A4 inhibition), Effect, Advice
3. Key categories, each with a short explanation: grapefruit family, vitamin K consistency, tyramine, potassium, sodium, St John's Wort, calcium/iron timing, CYP450 inducers and inhibitors
4. Medicine timing guide - what to take with food, on an empty stomach, or separated from which foods
5. Kenyan foods and herbal remedies to watch
6. Red-flag symptoms that need medical attention
7. Sources`,
    },
    patientGroup: {
        label: "Patient-group focus",
        description: "An in-depth report on the one patient group or condition named in the topic.",
        structure: `This article should focus on the single patient group or condition named in the topic, structured as:
1. Overview - the condition, why these patients are immunocompromised, and nutritional risks (1-2 paragraphs)
2. Assessment - what to check before planning a diet (age, weight trend, labs, renal/hepatic function, swallowing, allergies, medicines and supplements, food access and cost)
3. Nutritional requirements - energy, protein, micronutrients, fluids
4. Foods to favour and foods to avoid - a Markdown table, using affordable Kenyan foods
5. Drug-nutrient interactions relevant to this group - a Markdown table
6. Sample one-day Kenyan meal plan - a Markdown table
7. Monitoring and when to refer
8. Sources`,
    },
    summary: {
        label: "Quick summary",
        description: "A short one-page brief with the key points and sources.",
        structure: `This article should be a short one-page brief (about 400 words), structured as:
1. Key points - 5 to 8 bullet points
2. Foods to favour / foods to avoid - a short two-column Markdown table, using Kenyan foods where possible
3. Safety warnings - bullet points on the most important drug-nutrient interactions
4. Sources`,
    },
} satisfies Record<string, ReportType>

export type ReportTypeKey = keyof typeof REPORT_TYPES

export function isReportType(value: unknown): value is ReportTypeKey {
    return typeof value === "string" && Object.hasOwn(REPORT_TYPES, value)
}

// ---------------------------------------------------------------------------
// The graph
// ---------------------------------------------------------------------------

//define the memory schema for the report graph
const AgentState = Annotation.Root({
    topic: Annotation<string>(), //input provided by the user
    reportStructure: Annotation<string>(), //the layout the report must follow
    researchData: Annotation<string[]>(), //saved by Researcher, read by Author
    report: Annotation<string>(), //saved by the author as the final output
})

// The most searches the report researcher may run, including the Kenyan
// one. A report covers many angles, so it gets far more than a chat message.
const REPORT_MAX_SEARCHES = 12

/**
 * LangGraph node that gathers research context for the report topic, using
 * the ReAct researcher (`research` in agent.ts).
 */
async function researcherNode(state: typeof AgentState.State) {
    console.log("--- RUNNING RESEARCHER PHASE ---")

    const instructions = `You are the researcher for a clinical nutrition report on immunocompromised patients. You find the evidence; someone else writes the report. You never write the report yourself.

You have two search tools: ${WEB_SEARCH_TOOL} (the whole web) and ${KENYAN_SEARCH_TOOL} (Kenyan government and research sites only). You can run up to ${REPORT_MAX_SEARCHES - 1} searches in total.

How to work:
- Read the topic and the report structure, and work out which angles need evidence. Every section of the structure needs some.
- Possible angles (use the ones the topic and structure need): nutrition for elderly patients; patients on Revlimid (lenalidomide), Velcade (bortezomib) and dexamethasone; HIV patients; diabetes patients; cancer patients; infants and neonates; post-operative patients; food safety for neutropenic patients; renal and hepatic dietary restrictions; drug-nutrient interactions (grapefruit, vitamin K, tyramine, potassium, St John's Wort, calcium/iron timing, CYP450 inducers and inhibitors); and affordable Kenyan foods and meals.
- Search for the most important angles first. You may ask for several searches at once.
- Read the results of every search before deciding what to do next. If a result is off-topic, too general, or leaves out something the report needs, search again with a more specific query.
- Each query must cover a different angle and make sense on its own. Prefer authoritative sources (clinical guidelines, government bodies, academic papers).
- Use ${KENYAN_SEARCH_TOOL} for Kenyan foods, meals and local guidance.
- When every section has enough evidence, stop: reply with the single word DONE and do not call a tool.`

    const task = `Report topic: ${state.topic}

Report structure:
${state.reportStructure}`

    const researchData = await research(
        instructions,
        [{ role: "user", content: task }],
        REPORT_MAX_SEARCHES,
        state.topic
    )

    return { researchData }
}

/**
 * LangGraph node that turns the gathered research into the final Markdown
 * report, following `state.reportStructure`.
 */
async function authorNode(state: typeof AgentState.State) {
    console.log("--- RUNNING AUTHOR PHASE ---")

    const context = state.researchData.join("\n\n") || "(no search results were found)"

    const prompt = `You are a careful clinical nutrition writer. Using the research context below, write a professional report about: ${state.topic}

The readers care for immunocompromised patients in Kenya. Requirements:
- Use clear Markdown headings, bullet points and tables where the structure asks for them, in an objective tone. No emojis.
- Suggest affordable, easy-to-find Kenyan foods and meals.
- Consider age, medical history, allergies, renal/hepatic function, swallowing ability, labs and weight trend, full medicine and supplement list, food access and cost.
- Safety is key. Cover drug-nutrient interactions where relevant: grapefruit family, vitamin K consistency, tyramine, potassium, sodium, St John's Wort, calcium/iron timing, CYP450 inducers and inhibitors.
- Base every claim on the context. If the context does not cover something the structure asks for, say briefly that no reliable source was found for it rather than inventing details.
- Each context entry starts with its title and a "Link:" line. Entries that start with "${KENYAN_LABEL}" come from Kenyan government or research bodies; prefer them for Kenyan foods and meals, but never copy "${KENYAN_LABEL}" into the report.
- In the Sources section, list every source you used as a Markdown link: [Title](https://link). Only use links that appear in the context; never invent one.
- End with a one-line disclaimer that this is general information and diet or medication changes should be confirmed with a doctor, pharmacist or dietitian.

REPORT STRUCTURE (follow this layout exactly, section by section):
${state.reportStructure}

CONTEXT:
${context}

REPORT:`

    const response = await model.invoke(prompt)

    // content can be a string or a list of content parts; keep only the text.
    const report = typeof response.content === "string"
        ? response.content
        : response.content.map((part) => ("text" in part && typeof part.text === "string" ? part.text : "")).join("")

    return { report }
}

const workflow = new StateGraph(AgentState)
    .addNode("researcher", researcherNode)
    .addNode("author", authorNode)
    .addEdge("__start__", "researcher")
    .addEdge("researcher", "author")
    .addEdge("author", END)

const reportApp = workflow.compile()

/**
 * Runs the report graph (researcher -> author) for one topic and returns the
 * finished report as Markdown.
 */
export async function generateReport(topic: string, reportType: ReportTypeKey = "comprehensive"): Promise<string> {
    const state = await reportApp.invoke({ topic, reportStructure: REPORT_TYPES[reportType].structure })
    return state.report
}

// ---------------------------------------------------------------------------
// Terminal use: npm run report -- "topic" [reportType]
// ---------------------------------------------------------------------------

async function runAgent() {
    const topic = process.argv[2] || "Nutrition for immunocompromised patients"
    const type = isReportType(process.argv[3]) ? process.argv[3] : "comprehensive"
    const report = await generateReport(topic, type)

    console.log("\n====== GENERATED REPORT ======\n")
    console.log(report)

    await writeFile("report.md", report)
    console.log("\nReport saved to report.md")

    const html = `<!doctype html>
<meta charset="utf-8">
<title>Report</title>
<body style="max-width: 850px; margin: 2rem auto; font-family: sans-serif">
${await marked.parse(report)}
</body>`
    await writeFile("report.html", html)
    console.log("Report saved to report.html")
}

// Only run from the terminal, not when server.ts imports this file.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    runAgent().catch((error) => {
        console.error("Report generation failed:", error)
        process.exit(1)
    })
}
