const API = '';
const TOKEN_KEY = 'agro_token';
const USER_KEY = 'agro_user';
const THEME_KEY = 'agro_theme';
const GUEST_USED_KEY = 'agro_guest_used';
const SESSION_META_KEY = 'agro_session_meta_v1';

function MenuIcon({ name }) {
  const paths = {
    share: (
      <>
        <path d="M9 3h6v6" />
        <path d="M8 10L15 3" />
        <path d="M13 9v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h5" />
      </>
    ),
    group: (
      <>
        <path d="M9 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
        <path d="M2.7 16a6.4 6.4 0 0 1 12.6 0" />
        <path d="M14 7a2.2 2.2 0 0 1 0 4.4" />
      </>
    ),
    rename: (
      <>
        <path d="M3 13.5V16h2.5l7.1-7.1-2.5-2.5L3 13.5Z" />
        <path d="M10.9 5.1l2.5 2.5" />
      </>
    ),
    pin: (
      <>
        <path d="M6 4h6l-1 4 3 3v1H2v-1l3-3-1-4Z" />
        <path d="M8 12v4" />
      </>
    ),
    archive: (
      <>
        <path d="M3 4h12v3H3V4Z" />
        <path d="M4 7h10v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7Z" />
        <path d="M7 10h4" />
      </>
    ),
    delete: (
      <>
        <path d="M3 5h12" />
        <path d="M6 5V3h4v2" />
        <path d="M5 5l1 10h6l1-10" />
        <path d="M8 8v5" />
        <path d="M10 8v5" />
      </>
    ),
  };
  return (
    <svg className="session-menu-icon" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      {paths[name]}
    </svg>
  );
}

function App() {
  const [theme, setTheme] = React.useState(localStorage.getItem(THEME_KEY) || 'dark');
  const [token, setToken] = React.useState(localStorage.getItem(TOKEN_KEY) || '');
  const [user, setUser] = React.useState(() => {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  });

  const [guestUsed, setGuestUsed] = React.useState(localStorage.getItem(GUEST_USED_KEY) === '1');
  const [sessions, setSessions] = React.useState([]);
  const [searchQuery, setSearchQuery] = React.useState('');
  const [searchHits, setSearchHits] = React.useState([]);
  const [searchLoading, setSearchLoading] = React.useState(false);
  const [currentSessionId, setCurrentSessionId] = React.useState(null);
  const [messages, setMessages] = React.useState([]);
  const [showHero, setShowHero] = React.useState(true);
  const [showAuthModal, setShowAuthModal] = React.useState(false);
  const [showSettingsModal, setShowSettingsModal] = React.useState(false);
  const [sharedMode, setSharedMode] = React.useState(false);
  const [toast, setToast] = React.useState('');
  const [sessionMenuOpenId, setSessionMenuOpenId] = React.useState(null);
  const [pinnedSessionIds, setPinnedSessionIds] = React.useState([]);
  const [archivedSessionIds, setArchivedSessionIds] = React.useState([]);
  const [sessionTitleOverrides, setSessionTitleOverrides] = React.useState({});

  const [prompt, setPrompt] = React.useState('');
  const [photoName, setPhotoName] = React.useState('не выбрано');
  const [calibrationName, setCalibrationName] = React.useState('нет');

  const [loginForm, setLoginForm] = React.useState({ email: '', password: '' });
  const [authError, setAuthError] = React.useState('');
  const [passwordForm, setPasswordForm] = React.useState({ old_password: '', new_password: '' });
  const [settingsError, setSettingsError] = React.useState('');

  const photoInputRef = React.useRef(null);
  const calibrationInputRef = React.useRef(null);
  const chatScrollRef = React.useRef(null);

  const userStorageScope = React.useMemo(() => {
    if (user?.id) return `user_${user.id}`;
    return token ? 'auth_unknown' : 'guest';
  }, [user, token]);

  React.useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  React.useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(''), 2400);
    return () => clearTimeout(t);
  }, [toast]);

  React.useEffect(() => {
    let parsed = {};
    try {
      parsed = JSON.parse(localStorage.getItem(SESSION_META_KEY) || '{}') || {};
    } catch {
      parsed = {};
    }
    const scoped = parsed[userStorageScope] || {};
    const pinned = Array.isArray(scoped.pinned) ? scoped.pinned.map((x) => Number(x)).filter((x) => Number.isFinite(x)) : [];
    const archived = Array.isArray(scoped.archived) ? scoped.archived.map((x) => Number(x)).filter((x) => Number.isFinite(x)) : [];
    const titles = scoped.titles && typeof scoped.titles === 'object' ? scoped.titles : {};
    setPinnedSessionIds(pinned);
    setArchivedSessionIds(archived);
    setSessionTitleOverrides(titles);
  }, [userStorageScope]);

  React.useEffect(() => {
    let parsed = {};
    try {
      parsed = JSON.parse(localStorage.getItem(SESSION_META_KEY) || '{}') || {};
    } catch {
      parsed = {};
    }
    parsed[userStorageScope] = {
      pinned: pinnedSessionIds,
      archived: archivedSessionIds,
      titles: sessionTitleOverrides,
    };
    localStorage.setItem(SESSION_META_KEY, JSON.stringify(parsed));
  }, [pinnedSessionIds, archivedSessionIds, sessionTitleOverrides, userStorageScope]);

  React.useEffect(() => {
    if (sessionMenuOpenId == null) return undefined;
    const onPointerDown = (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest('.session-more') || target.closest('.session-menu')) return;
      setSessionMenuOpenId(null);
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [sessionMenuOpenId]);

  const scrollToBottom = (smooth = false) => {
    const el = chatScrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
  };

  React.useEffect(() => {
    requestAnimationFrame(() => scrollToBottom(false));
  }, [messages.length]);

  const authHeaders = (json = false) => {
    const h = {};
    if (token) h['Authorization'] = `Bearer ${token}`;
    if (json) h['Content-Type'] = 'application/json';
    return h;
  };

  const toWebPath = (path) => {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    const normalized = path.replaceAll('\\', '/');
    if (normalized.startsWith('/outputs/') || normalized.startsWith('outputs/')) {
      return normalized.startsWith('/') ? normalized : `/${normalized}`;
    }
    const idx = normalized.toLowerCase().lastIndexOf('/outputs/');
    if (idx >= 0) return normalized.slice(idx);
    return '';
  };

  const fetchSessions = React.useCallback(async () => {
    if (!token) {
      setSessions([]);
      return;
    }
    const res = await fetch(`${API}/chat/sessions?limit=300`, { headers: authHeaders() });
    if (!res.ok) return;
    const rows = await res.json();
    setSessions(rows);
    if (!currentSessionId && rows.length) setCurrentSessionId(rows[0].id);
  }, [token, currentSessionId]);

  React.useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  React.useEffect(() => {
    if (!token) {
      setSearchHits([]);
      return;
    }
    const q = searchQuery.trim();
    if (!q) {
      setSearchHits([]);
      return;
    }

    let cancelled = false;
    setSearchLoading(true);
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API}/chat/search?query=${encodeURIComponent(q)}&limit=80`, {
          headers: authHeaders(),
        });
        if (!res.ok) {
          if (!cancelled) setSearchHits([]);
          return;
        }
        const payload = await res.json();
        if (!cancelled) setSearchHits(payload || []);
      } finally {
        if (!cancelled) setSearchLoading(false);
      }
    }, 280);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [searchQuery, token]);

  const openSession = async (sessionId) => {
    if (!token) {
      setShowAuthModal(true);
      return;
    }
    setSharedMode(false);
    setCurrentSessionId(sessionId);
    setShowHero(false);
    const res = await fetch(`${API}/chat/sessions/${sessionId}/messages?limit=500`, {
      headers: authHeaders(),
    });
    if (!res.ok) {
      setToast('Не удалось открыть чат');
      return;
    }
    const rows = await res.json();
    setMessages(
      rows.map((m) => ({
        role: m.role === 'assistant' ? 'ai' : 'user',
        text: m.content,
        files: null,
        image_preview: null,
        image_name: null,
      }))
    );
    setTimeout(() => scrollToBottom(false), 0);
  };

  React.useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const shared = params.get('share') || params.get('session');
    if (!shared) return;
    (async () => {
      const res = await fetch(`${API}/chat/share/${shared}`);
      if (!res.ok) {
        setToast('Не удалось открыть чат по ссылке');
        return;
      }
      const payload = await res.json();
      setSharedMode(true);
      setShowHero(false);
      setCurrentSessionId(Number(payload.session_id));
      setMessages(
        (payload.messages || []).map((m) => ({
          role: m.role === 'assistant' ? 'ai' : 'user',
          text: m.content,
          files: null,
          image_preview: null,
          image_name: null,
        }))
      );
    })();
  }, []);

  const saveAuth = (payload) => {
    setToken(payload.access_token);
    setUser(payload.user || null);
    localStorage.setItem(TOKEN_KEY, payload.access_token);
    localStorage.setItem(USER_KEY, JSON.stringify(payload.user || {}));
    setShowAuthModal(false);
    setAuthError('');
    setToast('Вход выполнен');
  };

  const loginOrRegister = async (mode) => {
    const endpoint = mode === 'register' ? '/auth/register' : '/auth/login';
    const res = await fetch(`${API}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(loginForm),
    });
    const payload = await res.json();
    if (!res.ok) {
      setAuthError(payload.detail || 'Ошибка авторизации');
      return;
    }
    saveAuth(payload);
    fetchSessions();
  };

  const logout = () => {
    setToken('');
    setUser(null);
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    setSessions([]);
    setSearchHits([]);
    setSearchQuery('');
    setCurrentSessionId(null);
    setMessages([]);
    setSharedMode(false);
    setShowHero(true);
  };

  const createNewChat = () => {
    setCurrentSessionId(null);
    setMessages([]);
    setSharedMode(false);
    setShowHero(true);
    setSearchQuery('');
    if (!token) setToast('Новый гостевой диалог создан');
  };

  const addMessage = (message) => setMessages((prev) => [...prev, message]);

  const sendMessage = async () => {
    const text = (prompt || '').trim();
    const photo = photoInputRef.current?.files?.[0] || null;
    const calibration = calibrationInputRef.current?.files?.[0] || null;

    if (!text && !photo) {
      setToast('Введите вопрос или прикрепите фото');
      return;
    }
    if (!token && guestUsed && photo) {
      setShowAuthModal(true);
      setToast('Пробный анализ по фото уже использован. Войдите для полного режима.');
      return;
    }

    setShowHero(false);
    setSharedMode(false);
    addMessage({
      role: 'user',
      text: text || 'Проанализируй прикрепленное изображение растения.',
      files: null,
      image_preview: photo ? URL.createObjectURL(photo) : null,
      image_name: photo?.name || null,
    });
    addMessage({ role: 'ai', text: 'Обрабатываю запрос...', files: null, temp: true });

    try {
      if (photo) {
        const fd = new FormData();
        fd.append('image', photo);
        fd.append('message', text || 'Проведи диагностику растения по фото и дай рекомендации.');
        fd.append('crop', 'Unknown');
        fd.append('source_type', 'lab_camera');
        fd.append('camera_id', 'default');
        fd.append('conf', '0.08');
        fd.append('iou', '0.45');
        fd.append('max_det', '60');
        if (currentSessionId) fd.append('session_id', String(currentSessionId));
        if (calibration) fd.append('calibration_image', calibration);

        const res = await fetch(`${API}/chat/analyze`, {
          method: 'POST',
          headers: authHeaders(),
          body: fd,
        });
        const payload = await res.json();
        setMessages((prev) => prev.filter((m) => !m.temp));

        if (!res.ok) {
          addMessage({ role: 'ai', text: `Ошибка: ${payload.detail || 'не удалось выполнить запрос'}`, files: null });
          return;
        }

        setCurrentSessionId(payload.session_id || currentSessionId);
        addMessage({
          role: 'ai',
          text: payload.assistant_reply || 'Ответ не получен.',
          files: {
            overlay: toWebPath(payload.result?.files?.overlay || ''),
            csv: toWebPath(payload.result?.files?.csv || ''),
            pdf_report: toWebPath(payload.result?.files?.pdf_report || ''),
            distribution_plot: toWebPath(payload.result?.files?.distribution_plot || ''),
          },
          image_preview: null,
          image_name: null,
        });

        if (!token) {
          setGuestUsed(true);
          localStorage.setItem(GUEST_USED_KEY, '1');
        } else {
          fetchSessions();
        }
      } else {
        const res = await fetch(`${API}/chat/message`, {
          method: 'POST',
          headers: authHeaders(true),
          body: JSON.stringify({ message: text, session_id: currentSessionId }),
        });
        const payload = await res.json();
        setMessages((prev) => prev.filter((m) => !m.temp));

        if (!res.ok) {
          addMessage({ role: 'ai', text: `Ошибка: ${payload.detail || 'не удалось выполнить запрос'}`, files: null });
          return;
        }

        if (payload.session_id) setCurrentSessionId(payload.session_id);
        addMessage({ role: 'ai', text: payload.assistant_reply || 'Ответ не получен.', files: null });
        if (token) fetchSessions();
      }
    } finally {
      setPrompt('');
      if (photoInputRef.current) photoInputRef.current.value = '';
      if (calibrationInputRef.current) calibrationInputRef.current.value = '';
      setPhotoName('не выбрано');
      setCalibrationName('нет');
      setTimeout(() => scrollToBottom(true), 20);
    }
  };

  const copyShare = async () => {
    if (!currentSessionId) {
      setToast('Нет активного чата');
      return;
    }
    const url = `${window.location.origin}${window.location.pathname}?share=${currentSessionId}`;
    try {
      await navigator.clipboard.writeText(url);
      setToast('Ссылка скопирована');
    } catch {
      setToast(url);
    }
  };

  const copyShareForSession = async (sessionId) => {
    if (!sessionId) return;
    const url = `${window.location.origin}${window.location.pathname}?share=${sessionId}`;
    try {
      await navigator.clipboard.writeText(url);
      setToast('Ссылка чата скопирована');
    } catch {
      setToast(url);
    }
  };

  const renameSessionLocal = (sessionId, currentTitle) => {
    const nextTitle = window.prompt('Новое название чата', currentTitle || '');
    if (nextTitle == null) return;
    const cleanTitle = nextTitle.trim();
    if (!cleanTitle) {
      setToast('Название не может быть пустым');
      return;
    }

    setSessionTitleOverrides((prev) => ({ ...prev, [sessionId]: cleanTitle }));
    setSessions((prev) =>
      prev.map((row) => {
        const rowId = Number(row.session_id || row.id);
        if (rowId !== Number(sessionId)) return row;
        return { ...row, title: cleanTitle };
      })
    );
    setToast('Чат переименован');
  };

  const togglePinnedSession = (sessionId) => {
    const numericId = Number(sessionId);
    setPinnedSessionIds((prev) => {
      if (prev.includes(numericId)) return prev.filter((id) => id !== numericId);
      return [...prev, numericId];
    });
  };

  const archiveSessionLocal = (sessionId) => {
    const numericId = Number(sessionId);
    setArchivedSessionIds((prev) => (prev.includes(numericId) ? prev : [...prev, numericId]));
    setPinnedSessionIds((prev) => prev.filter((id) => id !== numericId));
    if (currentSessionId === numericId) createNewChat();
    setToast('Чат отправлен в архив');
  };

  const deleteSessionLocal = (sessionId) => {
    const numericId = Number(sessionId);
    const ok = window.confirm('Удалить чат из списка? Это действие можно отменить только вручную через localStorage.');
    if (!ok) return;
    setArchivedSessionIds((prev) => prev.filter((id) => id !== numericId));
    setPinnedSessionIds((prev) => prev.filter((id) => id !== numericId));
    setSessionTitleOverrides((prev) => {
      const next = { ...prev };
      delete next[numericId];
      return next;
    });
    setSessions((prev) => prev.filter((row) => Number(row.session_id || row.id) !== numericId));
    setSearchHits((prev) => prev.filter((row) => Number(row.session_id || row.id) !== numericId));
    if (currentSessionId === numericId) createNewChat();
    setToast('Чат удален из списка');
  };

  const changePassword = async () => {
    setSettingsError('');
    if (!token) {
      setSettingsError('Сначала войдите в аккаунт');
      return;
    }
    const res = await fetch(`${API}/auth/change-password`, {
      method: 'POST',
      headers: authHeaders(true),
      body: JSON.stringify(passwordForm),
    });
    const payload = await res.json();
    if (!res.ok) {
      setSettingsError(payload.detail || 'Не удалось сменить пароль');
      return;
    }
    setPasswordForm({ old_password: '', new_password: '' });
    setToast('Пароль обновлен');
  };

  const visibleSessionsRaw = (searchQuery.trim() ? searchHits : sessions).filter(
    (row) => !archivedSessionIds.includes(Number(row.session_id || row.id))
  );
  const visibleSessions = React.useMemo(() => {
    const rows = [...visibleSessionsRaw];
    rows.sort((a, b) => {
      const aId = Number(a.session_id || a.id);
      const bId = Number(b.session_id || b.id);
      const aPinned = pinnedSessionIds.includes(aId) ? 1 : 0;
      const bPinned = pinnedSessionIds.includes(bId) ? 1 : 0;
      if (aPinned !== bPinned) return bPinned - aPinned;
      return 0;
    });
    return rows;
  }, [visibleSessionsRaw, pinnedSessionIds]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <img src="/frontend/assets/plantvision-logo-clean.png" alt="PlantVision AI" />
        </div>

        <button className="btn primary new-chat-btn" onClick={createNewChat}>+ Новый чат</button>

        {token && (
          <div className="search-wrap">
            <input
              className="field"
              placeholder="Поиск по словам в чатах..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <div className="hint">{searchLoading ? 'Ищу...' : searchQuery.trim() ? `Найдено: ${searchHits.length}` : ' '}</div>
          </div>
        )}

        {!token && (
          <button
            className="mini-link"
            onClick={() => {
              logout();
              localStorage.removeItem(GUEST_USED_KEY);
              setGuestUsed(false);
              setToast('Гостевой режим включен');
            }}
          >
            Попробовать без регистрации
          </button>
        )}

        <div className="session-list">
          {visibleSessions.map((s) => {
            const sessionId = Number(s.session_id || s.id);
            const title = sessionTitleOverrides[sessionId] || s.title || ('Чат #' + sessionId);
            const subtitle = s.excerpt || (s.updated_at ? new Date(s.updated_at).toLocaleString() : '');
            const isPinned = pinnedSessionIds.includes(sessionId);
            return (
              <div key={`${sessionId}-${s.message_id || 'base'}`} className={`session-row ${currentSessionId === sessionId ? 'active' : ''}`}>
                <button
                  className={`session-item ${currentSessionId === sessionId ? 'active' : ''}`}
                  onClick={() => {
                    setSessionMenuOpenId(null);
                    openSession(sessionId);
                  }}
                >
                  <div className="session-title">{isPinned ? '📌 ' : ''}{title}</div>
                  <div className="session-sub">{subtitle}</div>
                </button>

                <button
                  className="session-more"
                  aria-label="Действия с чатом"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSessionMenuOpenId((prev) => (prev === sessionId ? null : sessionId));
                  }}
                >
                  ⋯
                </button>

                {sessionMenuOpenId === sessionId && (
                  <div className="session-menu" onClick={(e) => e.stopPropagation()} role="menu">
                    <button
                      className="session-menu-item"
                      onClick={() => {
                        copyShareForSession(sessionId);
                        setSessionMenuOpenId(null);
                      }}
                    >
                      <MenuIcon name="share" />
                      <span>Поделиться</span>
                    </button>
                    <button
                      className="session-menu-item"
                      onClick={() => {
                        setSessionMenuOpenId(null);
                        setToast('Групповые чаты появятся в следующем релизе');
                      }}
                    >
                      <MenuIcon name="group" />
                      <span>Начать групповой чат</span>
                    </button>
                    <button
                      className="session-menu-item"
                      onClick={() => {
                        renameSessionLocal(sessionId, title);
                        setSessionMenuOpenId(null);
                      }}
                    >
                      <MenuIcon name="rename" />
                      <span>Переименовать</span>
                    </button>
                    <div className="session-menu-divider" />
                    <button
                      className="session-menu-item"
                      onClick={() => {
                        togglePinnedSession(sessionId);
                        setSessionMenuOpenId(null);
                      }}
                    >
                      <MenuIcon name="pin" />
                      <span>{isPinned ? 'Открепить чат' : 'Закрепить чат'}</span>
                    </button>
                    <button
                      className="session-menu-item"
                      onClick={() => {
                        archiveSessionLocal(sessionId);
                        setSessionMenuOpenId(null);
                      }}
                    >
                      <MenuIcon name="archive" />
                      <span>Архивировать</span>
                    </button>
                    <button
                      className="session-menu-item danger"
                      onClick={() => {
                        deleteSessionLocal(sessionId);
                        setSessionMenuOpenId(null);
                      }}
                    >
                      <MenuIcon name="delete" />
                      <span>Удалить</span>
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {!token && (
          <div className="sidebar-note">
            Без входа доступен 1 пробный анализ по фото. Текстовая консультация работает, но история чатов,
            поиск и полный функционал доступны после регистрации.
          </div>
        )}
      </aside>

      <main className="main">
        <div className="topbar">
          <button className="pill" onClick={copyShare}>Поделиться</button>
          <button className="pill" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
            Тема: {theme === 'dark' ? 'темная' : 'светлая'}
          </button>
          <button className="pill" onClick={() => setShowSettingsModal(true)}>Настройки</button>
          <button className="pill" onClick={() => (token ? logout() : setShowAuthModal(true))}>
            {token ? user?.email || 'Профиль' : 'Войти'}
          </button>
        </div>

        <section className="chat-scroll" ref={chatScrollRef}>
          {sharedMode && (
            <div className="shared-banner">
              Режим общей ссылки: этот чат открыт по ссылке. Чтобы продолжить диалог в своем аккаунте, нажмите "Войти".
            </div>
          )}

          {showHero && (
            <div className="hero">
              <h1>Готов, когда ты готов.</h1>
              <div>
                Можно отправить фото растения для AI-анализа или просто задать вопрос текстом и получить понятную консультацию.
              </div>
            </div>
          )}

          <div className="messages">
            {messages.map((m, idx) => (
              <div key={idx} className={`row ${m.role}`}>
                <div className="bubble">
                  {m.image_name && <div className="attached-name">Прикреплено: {m.image_name}</div>}
                  {m.image_preview && <img className="attached-image" src={m.image_preview} alt={m.image_name || 'прикрепленное фото'} />}
                  {m.text}
                  {m.files && (
                    <div className="file-links">
                      {Object.entries(m.files)
                        .filter(([, v]) => !!v)
                        .map(([k, v]) => (
                          <a key={k} className="file-link" href={v} target="_blank" rel="noreferrer">
                            {k}
                          </a>
                        ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="composer-wrap">
          <div className="composer">
            <div className="chips">
              <button className="chip" onClick={() => photoInputRef.current?.click()}>Фото</button>
              <button className="chip" onClick={() => calibrationInputRef.current?.click()}>Калибровка</button>
              {(!token && guestUsed) && <div className="chip passive">Пробный анализ по фото исчерпан</div>}
            </div>

            <input
              ref={photoInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => setPhotoName(e.target.files?.[0]?.name || 'не выбрано')}
            />
            <input
              ref={calibrationInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => setCalibrationName(e.target.files?.[0]?.name || 'нет')}
            />

            <div className="input-row">
              <textarea
                value={prompt}
                placeholder="Спросите о состоянии растения. Можно отправить только текст или текст + фото."
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
              />
              <button className="btn primary" onClick={sendMessage}>Отправить</button>
            </div>

            <div className="composer-hint">Фото: {photoName} | Калибровка: {calibrationName}</div>
          </div>
        </section>
      </main>

      {showAuthModal && (
        <div className="modal-bg" onClick={() => setShowAuthModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Вход / Регистрация</h3>
            <div className="modal-row">
              <input
                className="field"
                type="email"
                placeholder="Эл. почта"
                value={loginForm.email}
                onChange={(e) => setLoginForm((v) => ({ ...v, email: e.target.value }))}
              />
              <input
                className="field"
                type="password"
                placeholder="Пароль"
                value={loginForm.password}
                onChange={(e) => setLoginForm((v) => ({ ...v, password: e.target.value }))}
              />
              {authError && <div className="error">{authError}</div>}
            </div>
            <div className="modal-actions">
              <button className="btn" onClick={() => loginOrRegister('login')}>Войти</button>
              <button className="btn primary" onClick={() => loginOrRegister('register')}>Регистрация</button>
            </div>
          </div>
        </div>
      )}

      {showSettingsModal && (
        <div className="modal-bg" onClick={() => setShowSettingsModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Настройки</h3>
            <div className="modal-row">
              <input
                className="field"
                type="password"
                placeholder="текущий пароль"
                value={passwordForm.old_password}
                onChange={(e) => setPasswordForm((v) => ({ ...v, old_password: e.target.value }))}
              />
              <input
                className="field"
                type="password"
                placeholder="новый пароль"
                value={passwordForm.new_password}
                onChange={(e) => setPasswordForm((v) => ({ ...v, new_password: e.target.value }))}
              />
              {settingsError && <div className="error">{settingsError}</div>}
            </div>
            <div className="modal-actions">
              <button className="btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>Сменить тему</button>
              <button className="btn" onClick={copyShare}>Копировать ссылку</button>
              <button className="btn primary" onClick={changePassword}>Сменить пароль</button>
            </div>
          </div>
        </div>
      )}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

if (!window.React || !window.ReactDOM) {
  document.getElementById('root').innerHTML =
    '<div style=\"padding:20px;font-family:Segoe UI,sans-serif\">Не удалось загрузить React runtime (проверьте интернет и блокировщик скриптов).</div>';
} else {
  ReactDOM.createRoot(document.getElementById('root')).render(<App />);
}
