import { useCallback, useState, type Dispatch, type SetStateAction } from 'react';
import { rebaseConfigDraft, type ConfigDraft } from '../services/configCoordinator';

export function useConfigDraft<T>(source: T | null, initial: T) {
    const [state, setState] = useState<ConfigDraft<T>>({ base: source, draft: source ?? initial, conflict: null });
    const [seen, setSeen] = useState(source);
    // Reconcile before rendering children so a save can never use the previous
    // revision while the canonical config has already advanced.
    let current = state;
    if (source !== seen) {
        setSeen(source);
        if (source !== null) {
            current = rebaseConfigDraft(state, source);
            setState(current);
        }
    }
    const setDraft: Dispatch<SetStateAction<T>> = useCallback(value => {
        setState(previous => ({ ...previous, draft: typeof value === 'function'
            ? (value as (previous: T) => T)(previous.draft) : value }));
    }, []);
    const reset = () => {
        if (source !== null) setState({ base: source, draft: source, conflict: null });
    };
    return [current.draft, setDraft, current.conflict, reset] as const;
}
