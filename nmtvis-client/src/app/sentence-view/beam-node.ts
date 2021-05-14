export interface BeamNode {
    id: string;
    logprob: number;
    name: string;
    children?: BeamNode[];
}