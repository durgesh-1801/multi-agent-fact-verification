import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { X, Settings, Server, Check, Activity, Cpu } from "lucide-react";
import { fetchHealthCheck } from "../api";
import { API_BASE_URL } from "../api/client";
import { useAnalysisContext } from "../context/AnalysisContext";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const { selectedModelProvider, setSelectedModelProvider } = useAnalysisContext();
  const [autoSave, setAutoSave] = useState(true);
  const [latexFormatting, setLatexFormatting] = useState(true);
  const [strictnessLevel, setStrictnessLevel] = useState("high");
  const [saved, setSaved] = useState(false);

  // TanStack Query for backend GET /api/v1/health status
  const { data: healthData, isLoading: healthLoading, isError: healthError } = useQuery({
    queryKey: ["healthCheck"],
    queryFn: fetchHealthCheck,
    enabled: isOpen, // Only query when settings modal is opened
    refetchInterval: 15000, // Poll every 15 seconds
  });

  if (!isOpen) return null;

  const handleSaveSettings = () => {
    setSaved(true);
    setTimeout(() => {
      setSaved(false);
      onClose();
    }, 800);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-md flex items-center justify-center p-4">
      <div className="glass-card max-w-lg w-full rounded-2xl p-6 border border-gray-200 bg-white space-y-6 relative animate-in fade-in zoom-in-95 duration-200 shadow-2xl">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 rounded-lg text-gray-400 hover:text-gray-900 hover:bg-gray-100"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 border-b border-gray-200 pb-4">
          <Settings className="w-6 h-6 text-[#4F46E5]" />
          <div>
            <h2 className="text-xl font-bold text-gray-900">VeriSphere Configuration</h2>
            <p className="text-xs text-gray-500">Manage multi-agent LLM parameters and backend API integration.</p>
          </div>
        </div>

        <div className="space-y-4">
          {/* Server Connection Status via GET /api/v1/health */}
          <div className="p-3.5 rounded-xl bg-gray-50 border border-gray-200 space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <Server className="w-4 h-4 text-[#06B6D4]" />
                <div>
                  <p className="font-bold text-gray-900">FastAPI Backend (v1 API)</p>
                  <p className="text-gray-500 font-mono">{API_BASE_URL}/api/v1</p>
                </div>
              </div>
              <span
                className={`px-2.5 py-0.5 rounded-md font-bold text-[11px] border ${
                  healthData && !healthError
                    ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                    : healthLoading
                    ? "bg-amber-50 text-amber-700 border-amber-200"
                    : "bg-rose-50 text-rose-700 border-rose-200"
                }`}
              >
                {healthData && !healthError
                  ? "CONNECTED"
                  : healthLoading
                  ? "CHECKING..."
                  : "OFFLINE (FALLBACK ACTIVE)"}
              </span>
            </div>

            {healthData && (
              <div className="pt-2 border-t border-gray-200 flex items-center justify-between font-mono text-[11px] text-gray-600">
                <span>MongoDB: <strong className="text-emerald-700">{healthData.mongodb}</strong></span>
                <span>Status: <strong className="text-emerald-700">{healthData.status}</strong></span>
              </div>
            )}
          </div>

          {/* Model Provider Selection */}
          <div className="p-3.5 rounded-xl bg-gray-50 border border-gray-200 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-gray-900">
              <Cpu className="w-4 h-4 text-[#4F46E5]" />
              <span>Primary LLM Provider Engine</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {[
                { id: "gemini", label: "Gemini 2.5 Flash" },
                { id: "claude", label: "Claude 3.5 Sonnet" },
                { id: "openai", label: "GPT-4o" },
              ].map((prov) => (
                <button
                  key={prov.id}
                  onClick={() => setSelectedModelProvider(prov.id)}
                  className={`py-2 px-2 rounded-lg text-xs font-bold transition-all border ${
                    selectedModelProvider === prov.id
                      ? "bg-[#4F46E5] text-white border-indigo-600 shadow-xs"
                      : "bg-white text-gray-700 border-gray-200 hover:bg-gray-100"
                  }`}
                >
                  {prov.label}
                </button>
              ))}
            </div>
          </div>

          {/* Settings Options */}
          <div className="space-y-3">
            <label className="flex items-center justify-between p-3 rounded-xl bg-gray-50 border border-gray-200 cursor-pointer">
              <span className="text-xs font-semibold text-gray-900">Auto-Archive Verified Reports to Vault</span>
              <input
                type="checkbox"
                checked={autoSave}
                onChange={(e) => setAutoSave(e.target.checked)}
                className="w-4 h-4 accent-[#4F46E5]"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-xl bg-gray-50 border border-gray-200 cursor-pointer">
              <span className="text-xs font-semibold text-gray-900">Enable Mathematical & LaTeX Inline Formatting</span>
              <input
                type="checkbox"
                checked={latexFormatting}
                onChange={(e) => setLatexFormatting(e.target.checked)}
                className="w-4 h-4 accent-[#4F46E5]"
              />
            </label>

            <div className="p-3.5 rounded-xl bg-gray-50 border border-gray-200 space-y-2">
              <span className="text-xs font-semibold text-gray-900 block">Default Verification Threshold</span>
              <div className="flex rounded-lg bg-gray-200 p-1 text-xs">
                {["balanced", "high", "extreme"].map((lvl) => (
                  <button
                    key={lvl}
                    onClick={() => setStrictnessLevel(lvl)}
                    className={`flex-1 py-1 font-semibold rounded-md capitalize transition-all ${
                      strictnessLevel === lvl ? "bg-[#4F46E5] text-white shadow-xs" : "text-gray-600 hover:text-gray-900"
                    }`}
                  >
                    {lvl}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="pt-4 border-t border-gray-200 flex items-center justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-100 text-gray-800 text-xs font-semibold rounded-xl hover:bg-gray-200 border border-gray-200"
          >
            Cancel
          </button>
          <button
            onClick={handleSaveSettings}
            className="px-5 py-2 bg-[#4F46E5] text-white text-xs font-semibold rounded-xl hover:bg-indigo-700 flex items-center gap-1.5"
          >
            {saved ? <Check className="w-4 h-4 text-emerald-300" /> : null}
            <span>{saved ? "Saved" : "Save Preferences"}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
