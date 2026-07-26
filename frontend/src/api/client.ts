import axios, { AxiosError, AxiosInstance, AxiosRequestConfig, AxiosResponse } from "axios";
import { getStoredToken, handleUnauthorizedRedirect } from "./auth";

// Centralized API Base URL configuration with fallback to http://localhost:8000
export const API_BASE_URL = (
  import.meta.env.VITE_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

/**
 * Centralized Axios Instance for VeriSphere API Communications
 */
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2-minute timeout for multi-agent LLM analysis runs
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
});

/**
 * Request Interceptor:
 * Logs endpoint requests and attaches Authorization Header (Bearer token) if JWT is present
 */
apiClient.interceptors.request.use(
  (config) => {
    const endpoint = config.url || "";
    console.log("Backend URL:", API_BASE_URL);
    console.log("Calling:", endpoint);

    const token = getStoredToken();
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

/**
 * Retry helper for transient network failures
 */
const shouldRetry = (error: AxiosError): boolean => {
  if (!error.response) {
    // Network error (e.g. ECONNREFUSED, offline)
    return true;
  }
  // Server error 502, 503, 504
  return [502, 503, 504].includes(error.response.status);
};

/**
 * Response Interceptor:
 * Global Error Handling, Retry Logic, and 401 Unauthorized handling
 */
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  async (error: AxiosError) => {
    const config = error.config as AxiosRequestConfig & { _retryCount?: number };

    // Automatic Retry logic for transient network issues (up to 2 retries)
    if (config && shouldRetry(error)) {
      config._retryCount = config._retryCount || 0;
      if (config._retryCount < 2) {
        config._retryCount += 1;
        const delay = config._retryCount * 1000;
        await new Promise((resolve) => setTimeout(resolve, delay));
        return apiClient(config);
      }
    }

    // Global Error Handling
    if (error.response) {
      const status = error.response.status;

      if (status === 401) {
        handleUnauthorizedRedirect();
      }

      const message =
        (error.response.data as any)?.message ||
        (error.response.data as any)?.detail ||
        `Server error: ${status}`;

      console.error(`[API Error ${status}]:`, message);
    } else if (error.request) {
      console.error("[API Network Error]: No response received from server at", API_BASE_URL);
    } else {
      console.error("[API Request Error]:", error.message);
    }

    return Promise.reject(error);
  }
);

export default apiClient;
