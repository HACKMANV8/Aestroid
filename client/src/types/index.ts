export interface ChatMessage {
  userMessage: string;
  botResponse: string;
  timestamp: Date;
}

export interface LocationAlert {
  message: string;
  unitId: string;
  unitType: string;
  latitude: number;
  longitude: number;
  timestamp: Date;
}

export interface Location {
  unitId: string;
  unitType: string;
  latitude: number;
  longitude: number;
  timestamp: Date;
}