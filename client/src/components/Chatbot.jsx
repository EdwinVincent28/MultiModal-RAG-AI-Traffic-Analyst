import { useState, useRef, useEffect, useCallback } from "react";
import {
	MessageSquare,
	X,
	Send,
	ChevronRight,
	Loader2,
	AlertCircle,
	Minimize2,
	Maximize2,
	Zap,
	Paperclip,
} from "lucide-react";

const CHAT_API_URL = "http://localhost:5000/api/chat";

const SUGGESTIONS = [
	"What is the current traffic density?",
	"How many vehicles have been detected?",
	"What is the vehicle type breakdown?",
	"Are there any anomalies in the flow rate?",
	"Summarise the traffic conditions right now",
];

function fileToBase64(file) {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result.split(",")[1]); // strip data:...;base64,
		reader.onerror = reject;
		reader.readAsDataURL(file);
	});
}

// API call — sends text messages + optional image as base64
async function callChatAPI(messages, imageFile = null) {
	const formData = new FormData();

	// Send only the latest user question to the backend
	const lastUserMessage = [...messages]
		.reverse()
		.find((m) => m.role === "user");
	formData.append("question", lastUserMessage?.content || "");

	if (imageFile) {
		formData.append("image", imageFile); // This matches upload.single('image')
	}

	const res = await fetch(CHAT_API_URL, {
		method: "POST",
		body: formData, // No headers needed, browser sets Content-Type to multipart/form-data
	});

	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		throw new Error(err.message || `API error ${res.status}`);
	}

	return await res.json(); // Returns { reply, evidenceImages }
}

// Typing animation dots
function TypingDots() {
	return (
		<div className="flex items-center gap-1.5 px-1 py-0.5">
			{[0, 150, 300].map((delay) => (
				<span
					key={delay}
					className="h-2 w-2 rounded-full bg-indigo-400 animate-bounce shadow-[0_0_6px_rgba(99,102,241,0.8)]"
					style={{ animationDelay: `${delay}ms` }}
				/>
			))}
		</div>
	);
}
// Single message bubble
function Message({ msg }) {
	const isUser = msg.role === "user";
	const isError = msg.role === "error";

	return (
		<div className={`flex gap-2.5 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
			{/* ... Avatar code stays same ... */}

			<div
				className={[
					"max-w-[86%] rounded-2xl px-3.5 py-2.5 text-[13px] leading-relaxed",
					isUser
						? "bg-indigo-600/30 text-indigo-50"
						: "bg-slate-800/60 text-slate-100",
				].join(" ")}
			>
				{/* 1. Show user's local upload preview */}
				{msg.previewUrl && (
					<img
						src={msg.previewUrl}
						className="mb-2 max-h-40 w-full rounded-xl object-cover"
					/>
				)}

				{/* 2. Show the AI's evidence image from the backend */}
				{msg.evidenceUrl && (
					<div className="mb-2 space-y-1">
						<p className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">
							Matched Vehicle
						</p>
						<img
							src={msg.evidenceUrl}
							alt="Evidence"
							className="max-h-48 w-full rounded-xl object-cover ring-1 ring-indigo-500/30 shadow-lg"
						/>
					</div>
				)}

				<p className="whitespace-pre-wrap">{msg.content}</p>
				{/* ... Timestamp stays same ... */}
			</div>
		</div>
	);
}

// MAIN CHATBOT COMPONENT
export default function Chatbot({ trafficContext = {} }) {
	const [open, setOpen] = useState(false);
	const [expanded, setExpanded] = useState(false);
	const [messages, setMessages] = useState([
		{
			role: "assistant",
			content:
				"Traffic AI online. Ask me anything about the current flow rate, vehicle counts, density, or anomalies.",
			ts: Date.now(),
		},
	]);
	const [input, setInput] = useState("");
	const [loading, setLoading] = useState(false);
	const bottomRef = useRef(null);
	const inputRef = useRef(null);
	const textareaRef = useRef(null);
	const fileInputRef = useRef(null);
	const [pendingImage, setPendingImage] = useState(null); // { file, previewUrl }

	// Scroll to bottom on new messages
	useEffect(() => {
		bottomRef.current?.scrollIntoView({ behavior: "smooth" });
	}, [messages, loading]);

	// Focus input when opened
	useEffect(() => {
		if (open) setTimeout(() => inputRef.current?.focus(), 120);
	}, [open]);

	// Auto-resize textarea
	const handleInput = (e) => {
		setInput(e.target.value);
		e.target.style.height = "auto";
		e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
	};

	const sendMessage = useCallback(
		async (text) => {
			const trimmed = (text ?? input).trim();
			if ((!trimmed && !pendingImage) || loading) return;

			const userMsg = {
				role: "user",
				content: trimmed,
				ts: Date.now(),
				previewUrl: pendingImage?.previewUrl ?? null,
			};

			const nextMessages = [...messages, userMsg];
			setMessages(nextMessages);
			setInput("");
			setLoading(true);

			const imageToSend = pendingImage?.file ?? null;
			setPendingImage(null);

			try {
                const data = await callChatAPI(nextMessages, imageToSend);

                let cleanReply = data.reply;
                let matchedEvidenceUrl = null;

                const idMatch = data.reply.match(/\[ID:\s*([a-fA-F0-9]{24})\]/);

                if (idMatch) {
                    const mongoId = idMatch[1];
                    
                    const exactImage = data.evidenceImages?.find(img => img.mongo_id === mongoId);
                    
                    if (exactImage) {
                        matchedEvidenceUrl = exactImage.url;
                    }
                    cleanReply = data.reply.replace(/\[ID:\s*[a-fA-F0-9]{24}\]/g, "").trim();
                } else if (data.evidenceImages?.length > 0 && !data.reply.includes("STATISTICS")) {
                    matchedEvidenceUrl = data.evidenceImages[0].url;
                }

                setMessages((prev) => [
                    ...prev,
                    {
                        role: "assistant",
                        content: cleanReply,
                        evidenceUrl: matchedEvidenceUrl,
                        ts: Date.now(),
                    },
                ]);
            } catch (err) {
				setMessages((prev) => [
					...prev,
					{ role: "error", content: err.message, ts: Date.now() },
				]);
			} finally {
				setLoading(false);
			}
		},
		[input, loading, messages, pendingImage],
	);

	const handleKey = (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			sendMessage();
		}
	};

	const clearChat = () => {
		setMessages([
			{
				role: "assistant",
				content:
					"Chat cleared. What would you like to know about the traffic data?",
				ts: Date.now(),
			},
		]);
	};

	const showSuggestions = messages.length <= 1 && !loading;

	// Panel dimensions
	const panelWidth = expanded ? "w-[520px]" : "w-[380px]";
	const panelHeight = expanded ? "h-[680px]" : "h-[540px]";

	return (
		<>
			{/* ── FAB (floating action button) ────────────────── */}
			<button
				onClick={() => setOpen((v) => !v)}
				aria-label="Toggle Traffic AI chat"
				className={[
					"fixed bottom-6 right-6 z-50 flex items-center gap-2.5 rounded-2xl px-4 py-3",
					"bg-gradient-to-br from-indigo-600 to-violet-700",
					"shadow-[0_4px_24px_rgba(99,102,241,0.5)] ring-1 ring-indigo-400/40",
					"transition-all duration-200 hover:scale-105 hover:shadow-[0_6px_32px_rgba(99,102,241,0.65)] active:scale-95",
					open ? "opacity-0 pointer-events-none" : "opacity-100",
				].join(" ")}
			>
				<div className="relative flex h-8 w-8 items-center justify-center rounded-xl bg-white/15">
					<MessageSquare size={16} className="text-white" />
					<span className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full bg-green-400 ring-2 ring-violet-800 shadow-[0_0_8px_#4ade80]" />
				</div>
				<span className="text-sm font-bold text-white tracking-wide">
					Traffic AI
				</span>
			</button>

			{/* ── Chat Panel ──────────────────────────────────── */}
			<div
				className={[
					"fixed bottom-6 right-6 z-50 flex flex-col",
					"rounded-2xl overflow-hidden",
					"bg-[#0f0f1a] ring-1 ring-indigo-500/25",
					"shadow-[0_8px_64px_rgba(99,102,241,0.15),0_2px_32px_rgba(0,0,0,0.9)]",
					panelWidth,
					panelHeight,
					"transition-all duration-300 ease-out",
					open
						? "opacity-100 translate-y-0 scale-100"
						: "opacity-0 translate-y-4 scale-95 pointer-events-none",
				].join(" ")}
			>
				{/* Header */}
				<div className="flex flex-shrink-0 items-center justify-between border-b border-indigo-500/20 bg-gradient-to-r from-[#0f0f1a] via-[#13102a] to-[#0f0f1a] px-4 py-3.5">
					<div className="flex items-center gap-3">
						<div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-400/20 ring-1 ring-indigo-400/60 shadow-[0_0_16px_rgba(99,102,241,0.4)]">
							<Zap size={15} className="text-indigo-300" />
							<span className="absolute -right-1 -top-1 h-2 w-2 rounded-full bg-green-400 ring-2 ring-[#0f0f1a] shadow-[0_0_8px_#4ade80]" />
						</div>
						<div>
							<p className="text-[13px] font-bold text-white leading-tight tracking-wide">
								Traffic AI
							</p>
							<p className="text-[10px] text-indigo-400/70 leading-tight tracking-widest uppercase">
								Natural language · Live data
							</p>
						</div>
					</div>

					<div className="flex items-center gap-1">
						{/* Clear */}
						<button
							onClick={clearChat}
							title="Clear chat"
							className="rounded-lg px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-slate-500 transition hover:bg-indigo-500/10 hover:text-indigo-300"
						>
							Clear
						</button>

						{/* Expand/shrink */}
						<button
							onClick={() => setExpanded((v) => !v)}
							className="flex h-7 w-7 items-center justify-center rounded-lg text-slate-500 transition hover:bg-indigo-500/10 hover:text-indigo-300"
						>
							{expanded ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
						</button>

						{/* Close */}
						<button
							onClick={() => setOpen(false)}
							className="flex h-7 w-7 items-center justify-center rounded-lg text-slate-500 transition hover:bg-red-500/10 hover:text-red-400"
						>
							<X size={14} />
						</button>
					</div>
				</div>

				{/* Messages */}
				<div className="flex-1 overflow-y-auto px-4 py-4 bg-[#0c0c18]">
					<div className="space-y-4">
						{messages.map((msg, i) => (
							<Message key={i} msg={msg} />
						))}

						{loading && (
							<div className="flex gap-2.5">
								<div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg bg-indigo-400/20 ring-1 ring-indigo-400/40 shadow-[0_0_10px_rgba(99,102,241,0.25)]">
									<Loader2 size={13} className="animate-spin text-indigo-300" />
								</div>
								<div className="rounded-2xl rounded-tl-sm bg-slate-800/50 px-3.5 py-2.5 ring-1 ring-slate-600/30">
									<TypingDots />
								</div>
							</div>
						)}

						<div ref={bottomRef} />
					</div>
				</div>

				{/* Suggestions */}
				{showSuggestions && (
					<div className="flex-shrink-0 space-y-1.5 border-t border-indigo-500/15 bg-[#0c0c18] px-4 py-3">
						<p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-indigo-400/50">
							Try asking
						</p>
						{SUGGESTIONS.map((s) => (
							<button
								key={s}
								onClick={() => sendMessage(s)}
								className="group flex w-full items-center justify-between rounded-xl border border-indigo-500/10 bg-indigo-950/30 px-3 py-2 text-left text-[12px] text-slate-400 transition hover:border-indigo-400/30 hover:bg-indigo-500/10 hover:text-indigo-200 hover:shadow-[0_0_12px_rgba(99,102,241,0.1)]"
							>
								<span>{s}</span>
								<ChevronRight
									size={12}
									className="flex-shrink-0 opacity-0 transition group-hover:opacity-70"
								/>
							</button>
						))}
					</div>
				)}

				{/* Input */}
				<div className="flex-shrink-0 border-t border-indigo-500/15 bg-[#0f0f1a] p-3">
					{/* Image preview strip */}
					{pendingImage && (
						<div className="mb-2 flex items-center gap-2 rounded-xl bg-indigo-950/40 px-3 py-2 ring-1 ring-indigo-500/20">
							<img
								src={pendingImage.previewUrl}
								alt="preview"
								className="h-10 w-10 rounded-lg object-cover ring-1 ring-indigo-400/30"
							/>
							<span className="flex-1 truncate text-[11px] text-slate-400">
								{pendingImage.file.name}
							</span>
							<button
								onClick={() => setPendingImage(null)}
								className="flex h-5 w-5 items-center justify-center rounded-full text-slate-500 hover:text-red-400 transition"
							>
								<X size={13} />
							</button>
						</div>
					)}

					{/* Hidden file input */}
					<input
						ref={fileInputRef}
						type="file"
						accept="image/*"
						className="hidden"
						onChange={(e) => {
							const file = e.target.files?.[0];
							if (!file) return;
							const previewUrl = URL.createObjectURL(file);
							setPendingImage({ file, previewUrl });
							e.target.value = ""; // reset so same file can be re-selected
						}}
					/>

					<div className="flex items-end gap-2">
						{/* Upload button */}
						<button
							onClick={() => fileInputRef.current?.click()}
							disabled={loading}
							title="Attach image"
							className={[
								"flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl transition",
								pendingImage
									? "bg-indigo-500/20 text-indigo-300 ring-1 ring-indigo-400/40"
									: "bg-white/5 text-slate-500 hover:bg-indigo-500/10 hover:text-indigo-300",
								"disabled:opacity-40 disabled:cursor-not-allowed",
							].join(" ")}
						>
							<Paperclip size={15} />
						</button>

						<textarea
							ref={(el) => {
								inputRef.current = el;
								textareaRef.current = el;
							}}
							rows={1}
							value={input}
							onChange={handleInput}
							onKeyDown={handleKey}
							disabled={loading}
							placeholder="Ask about traffic conditions…"
							className={[
								"flex-1 resize-none rounded-xl px-3.5 py-2.5",
								"bg-[#0d1520] text-[13px] text-slate-100 placeholder-slate-600",
								"ring-1 ring-indigo-500/20 transition",
								"focus:outline-none focus:ring-indigo-400/50 focus:shadow-[0_0_0_3px_rgba(99,102,241,0.1)]",
								"disabled:opacity-40",
							].join(" ")}
							style={{ minHeight: "44px", maxHeight: "120px" }}
						/>
						<button
							onClick={() => sendMessage()}
							disabled={!input.trim() || loading}
							className={[
								"flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl transition",
								input.trim() && !loading
									? "bg-gradient-to-br from-indigo-500 to-violet-600 text-white shadow-[0_0_20px_rgba(99,102,241,0.5)] hover:shadow-[0_0_28px_rgba(99,102,241,0.7)] hover:scale-105"
									: "bg-white/5 text-slate-600 cursor-not-allowed",
							].join(" ")}
						>
							<Send size={15} />
						</button>
					</div>
					<p className="mt-2 text-center text-[10px] text-slate-600">
						Enter to send · Shift+Enter for newline
					</p>
				</div>
			</div>
		</>
	);
}
