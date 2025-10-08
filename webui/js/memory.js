import { fetchApi } from "./api.js";

async function fetchJson(url, options) {
  const response = await fetchApi(url, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return await response.json();
}

async function postJson(url, payload) {
  return await fetchJson(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function memoryDashboard() {
  return {
    sessions: [],
    session: "default",
    stats: null,
    items: [],
    query: "",
    topK: 20,
    limit: 25,
    offset: 0,
    mode: "recent",
    isLoading: false,
    isReindexing: false,
    selectedIds: [],
    initializeAttempted: false,

    get hasSelection() {
      return this.selectedIds.length > 0;
    },

    get pageStart() {
      return this.items.length > 0 ? this.offset + 1 : 0;
    },

    get pageEnd() {
      return this.offset + this.items.length;
    },

    get totalCount() {
      return this.stats?.total_count ?? 0;
    },

    get canPrev() {
      return this.offset > 0;
    },

    get canNext() {
      return this.offset + this.items.length < (this.mode === "search" ? this.topK : this.totalCount);
    },

    async init() {
      if (this.initializeAttempted) return;
      this.initializeAttempted = true;
      try {
        await this.refreshSessions();
        await this.fetchStats();
        await this.loadRecent(true);
      } catch (error) {
        console.error("Failed to initialize memory dashboard:", error);
        window.toastFrontendError?.(
          `Unable to initialize memory dashboard: ${error.message}`,
          "Memory Dashboard"
        );
      }
    },

    async refreshSessions() {
      try {
        const data = await fetchJson("/runtime/admin/memory/sessions");
        const sessions = Array.isArray(data.sessions) ? data.sessions : [];
        this.sessions = sessions;
        if (!sessions.includes(this.session)) {
          this.session = sessions[0] || "default";
        }
      } catch (error) {
        console.error("Failed to load memory sessions:", error);
        window.toastFrontendError?.(
          `Unable to load memory sessions: ${error.message}`,
          "Memory Dashboard"
        );
      }
    },

    async fetchStats() {
      try {
        const data = await fetchJson("/runtime/admin/memory/stats");
        this.stats = data.stats || null;
      } catch (error) {
        console.error("Failed to load memory stats:", error);
        window.toastFrontendError?.(
          `Unable to load memory stats: ${error.message}`,
          "Memory Dashboard"
        );
      }
    },

    async loadRecent(resetOffset = false) {
      if (resetOffset) {
        this.offset = 0;
      }
      this.mode = "recent";
      await this._loadItems();
    },

    async searchMemories(resetOffset = false) {
      if (!this.query.trim()) {
        await this.loadRecent(resetOffset);
        return;
      }
      if (resetOffset) {
        this.offset = 0;
      }
      this.mode = "search";
      await this._loadItems();
    },

    async _loadItems() {
      this.isLoading = true;
      this.selectedIds = [];

      try {
        const params = new URLSearchParams();
        params.set("session", this.session);
        params.set("offset", `${Math.max(this.offset, 0)}`);

        if (this.mode === "search") {
          params.set("query", this.query.trim());
          params.set("top_k", `${Math.max(this.topK, 1)}`);
        } else {
          params.set("recent", `${Math.max(this.limit, 1)}`);
        }

        const data = await fetchJson(`/runtime/admin/memory?${params.toString()}`);
        this.items = Array.isArray(data.items) ? data.items : [];
        if (typeof data.total === "number") {
          this.stats = this.stats || {};
          this.stats.total_count = data.total;
        }
      } catch (error) {
        console.error("Failed to load memory items:", error);
        window.toastFrontendError?.(
          `Unable to load memories: ${error.message}`,
          "Memory Dashboard"
        );
      } finally {
        this.isLoading = false;
      }

      await this.fetchStats();
    },

    async nextPage() {
      if (!this.canNext) return;
      this.offset += this.mode === "search" ? this.topK : this.limit;
      await this._loadItems();
    },

    async prevPage() {
      if (!this.canPrev) return;
      const step = this.mode === "search" ? this.topK : this.limit;
      this.offset = Math.max(this.offset - step, 0);
      await this._loadItems();
    },

    toggleSelect(id) {
      if (!id) return;
      if (this.selectedIds.includes(id)) {
        this.selectedIds = this.selectedIds.filter((item) => item !== id);
      } else {
        this.selectedIds = [...this.selectedIds, id];
      }
    },

    isSelected(id) {
      return this.selectedIds.includes(id);
    },

    async deleteSelected() {
      if (!this.hasSelection) return;
      try {
        await postJson("/runtime/admin/memory/bulk-delete", {
          ids: this.selectedIds,
        });
        window.toastFrontendInfo?.(
          `Deleted ${this.selectedIds.length} memories`,
          "Memory Dashboard"
        );
        await this._afterMutation();
      } catch (error) {
        console.error("Failed to delete memories:", error);
        window.toastFrontendError?.(
          `Unable to delete memories: ${error.message}`,
          "Memory Dashboard"
        );
      }
    },

    async deleteItem(id) {
      if (!id) return;
      try {
        const response = await fetchApi(`/runtime/admin/memory/${encodeURIComponent(id)}`, {
          method: "DELETE",
        });
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || "Failed to delete memory item");
        }
        window.toastFrontendInfo?.("Memory deleted", "Memory Dashboard");
        await this._afterMutation();
      } catch (error) {
        console.error("Failed to delete memory:", error);
        window.toastFrontendError?.(
          `Unable to delete memory: ${error.message}`,
          "Memory Dashboard"
        );
      }
    },

    async reindex() {
      if (this.isReindexing) return;
      this.isReindexing = true;
      try {
        await postJson("/runtime/admin/memory/reindex", {});
        window.toastFrontendInfo?.("Reindex started", "Memory Dashboard");
      } catch (error) {
        console.error("Failed to request reindex:", error);
        window.toastFrontendError?.(
          `Unable to reindex memory: ${error.message}`,
          "Memory Dashboard"
        );
      } finally {
        this.isReindexing = false;
        await this.fetchStats();
      }
    },

    async changeSession() {
      this.offset = 0;
      await this._loadItems();
    },

    readableTimestamp(ts) {
      if (!ts) return "—";
      try {
        const date = new Date(ts);
        if (Number.isNaN(date.getTime())) return ts;
        return date.toLocaleString();
      } catch {
        return ts;
      }
    },

    ellipsize(text, length = 120) {
      if (!text) return "";
      return text.length > length ? `${text.slice(0, length)}…` : text;
    },

    async _afterMutation() {
      await this._loadItems();
      await this.fetchStats();
    },
  };
}

if (!globalThis.memoryDashboard) {
  globalThis.memoryDashboard = memoryDashboard;
}

const memoryModalProxy = {
  open() {
    const modalEl = globalThis.document?.getElementById("memoryModal");
    if (!modalEl) {
      console.error("Memory modal element not found");
      return;
    }
    const modalAD = globalThis.Alpine ? Alpine.$data(modalEl) : null;
    if (!modalAD) {
      console.error("Memory modal is not ready");
      return;
    }
    modalAD.openModal();
  },

  close() {
    const modalEl = globalThis.document?.getElementById("memoryModal");
    if (!modalEl) return;
    const modalAD = globalThis.Alpine ? Alpine.$data(modalEl) : null;
    if (modalAD?.closeModal) {
      modalAD.closeModal();
    }
  },
};

document.addEventListener("alpine:init", () => {
  Alpine.data("memoryModalProxy", () => ({
    isOpen: false,
    dashboard: memoryDashboard(),
    init() {
      this.$watch("isOpen", (open) => {
        if (open && !this.dashboard.initializeAttempted) {
          this.dashboard.init();
        } else if (open) {
          this.dashboard.fetchStats();
        }
      });
    },
    openModal() {
      if (!this.dashboard.initializeAttempted) {
        this.dashboard.init();
      } else {
        this.dashboard.loadRecent(true);
      }
      this.isOpen = true;
    },
    closeModal() {
      this.isOpen = false;
    },
    handleOverlay(event) {
      if (event.target === event.currentTarget) {
        this.closeModal();
      }
    },
  }));
});

globalThis.memoryModalProxy = globalThis.memoryModalProxy || memoryModalProxy;
