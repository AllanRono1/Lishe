import express from "express"
import { rateLimit } from "express-rate-limit"
import { generateReport, isReportType, REPORT_TYPES } from "./index.js"

const server = express()

// Behind a hosting proxy (Render etc.) this lets the rate limiter see real IPs.
server.set("trust proxy", 1)
server.use(express.json())
server.use(express.static("public"))

const MAX_TOPIC_LENGTH = 300

// A report runs up to 12 searches and several model calls, so keep the
// number per visitor low: 5 reports every 30 minutes.
const reportLimiter = rateLimit({
    windowMs: 30 * 60 * 1000,
    limit: 5,
    message: { error: "Too many reports requested. Please try again later." },
})

// The report types the page offers in its drop-down.
server.get("/api/report-types", (_req, res) => {
    res.json(Object.entries(REPORT_TYPES).map(([id, t]) => ({ id, label: t.label, description: t.description })))
})

function validateReportRequest(req: express.Request, res: express.Response, next: express.NextFunction) {
    const { topic, reportType } = req.body ?? {}

    if (typeof topic !== "string" || !topic.trim()) {
        res.status(400).json({ error: "Please enter a topic." })
        return
    }
    if (topic.length > MAX_TOPIC_LENGTH) {
        res.status(400).json({ error: `Please keep the topic under ${MAX_TOPIC_LENGTH} characters.` })
        return
    }
    if (!isReportType(reportType)) {
        res.status(400).json({ error: "Please choose a valid report type." })
        return
    }
    next()
}

server.post("/api/report", validateReportRequest, reportLimiter, async (req, res) => {
    const { topic, reportType } = req.body
    try {
        const report = await generateReport(topic.trim(), reportType)
        res.json({ report })
    } catch (error) {
        console.error("Report generation failed:", error)
        res.status(500).json({ error: "The report could not be generated right now. Please try again." })
    }
})

const port = process.env.PORT || 3000

server.listen(port, (error) => {
    if (error) {
        console.error(`Could not start the server on port ${port}:`, error.message)
        process.exit(1)
    }
    console.log(`Server running at http://localhost:${port}`)
})
